from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import get_feature_cols, load_processed_dataset, make_standard_splits
from .fusion import make_l2_outputs
from .metrics import evaluate_method
from .models import predict_agent_score_matrix, train_agent_models
from .partitioning import (
    partition_aap,
    partition_random,
    save_partition_description,
    select_shared_features,
)


def _evaluate_partition_mode(
    mode_name: str,
    groups: Dict[str, List[str]],
    cfg: ExperimentConfig,
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    test_df: pd.DataFrame,
    calib_threshold_df: pd.DataFrame,
    calib_report_df: pd.DataFrame,
    shared_features: List[str],
) -> tuple[pd.DataFrame, dict]:
    agents = train_agent_models(tr_df, groups, shared_features, cfg)
    joblib.dump(agents, cfg.step2_models_path / f"step2_agents_{mode_name.lower()}.joblib")

    agent_names, va_mat = predict_agent_score_matrix(agents, va_df)
    _, te_mat = predict_agent_score_matrix(agents, test_df)
    _, ca_thr_mat = predict_agent_score_matrix(agents, calib_threshold_df)
    _, ca_rep_mat = predict_agent_score_matrix(agents, calib_report_df)

    outputs = make_l2_outputs(
        va_mat=va_mat,
        te_mat=te_mat,
        ca_thr_mat=ca_thr_mat,
        ca_rep_mat=ca_rep_mat,
        y_va=va_df["y"].values,
        cfg=cfg,
    )

    rows = []
    weight_rows = []
    for method_name, payload in outputs.items():
        rows.append(
            evaluate_method(
                method=method_name,
                level="L2_OPT",
                partition_mode=mode_name,
                comm_cost_units=float(len(agent_names)),
                test_scores=payload["test"],
                test_y=test_df["y"].values,
                calib_threshold_scores=payload["calib_threshold"],
                calib_threshold_y=calib_threshold_df["y"].values,
                calib_report_scores=payload["calib_report"],
                calib_report_y=calib_report_df["y"].values,
                target_fpr=cfg.target_fpr,
            )
        )
        if "weights" in payload:
            for agent_name, weight in zip(agent_names, payload["weights"]):
                weight_rows.append(
                    {
                        "partition_mode": mode_name,
                        "fusion_method": method_name,
                        "agent_name": agent_name,
                        "weight": float(weight),
                    }
                )
        if "health" in payload:
            for agent_name, health in zip(agent_names, payload["health"]):
                weight_rows.append(
                    {
                        "partition_mode": mode_name,
                        "fusion_method": f"{method_name}_health",
                        "agent_name": agent_name,
                        "weight": float(health),
                    }
                )

    return pd.DataFrame(rows), {"weights": pd.DataFrame(weight_rows), "agents": agents}


def run_step2_l2_optimization(cfg: ExperimentConfig) -> pd.DataFrame:
    cfg.ensure_dirs()
    df = load_processed_dataset(cfg)
    tr_df, va_df, test_df, calib_threshold_df, calib_report_df = make_standard_splits(df, cfg)
    feature_cols = get_feature_cols(df)
    shared_features = select_shared_features(tr_df, feature_cols, cfg)
    private_features = [c for c in feature_cols if c not in shared_features]

    random_groups = partition_random(private_features, cfg.n_agents, cfg.random_state)
    aap_groups = partition_aap(tr_df, private_features, cfg.n_agents)

    save_partition_description(
        cfg.step2_models_path / "step2_partition_random.json",
        method_name="RANDOM",
        shared_features=shared_features,
        groups=random_groups,
    )
    save_partition_description(
        cfg.step2_models_path / "step2_partition_aap.json",
        method_name="AAP",
        shared_features=shared_features,
        groups=aap_groups,
    )

    random_df, random_meta = _evaluate_partition_mode(
        "RANDOM",
        random_groups,
        cfg,
        tr_df,
        va_df,
        test_df,
        calib_threshold_df,
        calib_report_df,
        shared_features,
    )
    aap_df, aap_meta = _evaluate_partition_mode(
        "AAP",
        aap_groups,
        cfg,
        tr_df,
        va_df,
        test_df,
        calib_threshold_df,
        calib_report_df,
        shared_features,
    )

    res_df = pd.concat([random_df, aap_df], ignore_index=True)
    res_df = res_df.sort_values(["partition_mode", "method"]).reset_index(drop=True)
    res_df.to_csv(cfg.step2_results_path / "l2_optimization_metrics.csv", index=False)

    weights_df = pd.concat([random_meta["weights"], aap_meta["weights"]], ignore_index=True)
    weights_df.to_csv(cfg.step2_results_path / "l2_attention_weights.csv", index=False)

    base_random_mean = res_df[
        (res_df["partition_mode"] == "RANDOM") & (res_df["method"] == "L2_mean")
    ].iloc[0]
    res_df["tpr_gain_vs_random_l2_mean"] = res_df["test_tpr_at_fpr1"] - float(base_random_mean["test_tpr_at_fpr1"])
    res_df["pr_gain_vs_random_l2_mean"] = res_df["test_pr_auc"] - float(base_random_mean["test_pr_auc"])

    within_rows = []
    for partition_mode in sorted(res_df["partition_mode"].unique()):
        sub = res_df[res_df["partition_mode"] == partition_mode].copy()
        base = sub[sub["method"] == "L2_mean"].iloc[0]
        for row in sub.itertuples(index=False):
            within_rows.append(
                {
                    "partition_mode": partition_mode,
                    "method": row.method,
                    "tpr_gain_vs_same_partition_l2_mean": float(row.test_tpr_at_fpr1 - base.test_tpr_at_fpr1),
                    "pr_gain_vs_same_partition_l2_mean": float(row.test_pr_auc - base.test_pr_auc),
                }
            )
    gain_df = pd.DataFrame(within_rows)
    gain_df.to_csv(cfg.step2_results_path / "l2_gain_breakdown.csv", index=False)

    aap_gain_rows = []
    for method_name in sorted(res_df["method"].unique()):
        sub = res_df[res_df["method"] == method_name].set_index("partition_mode")
        if {"AAP", "RANDOM"}.issubset(sub.index):
            aap_gain_rows.append(
                {
                    "method": method_name,
                    "tpr_gain_aap_minus_random": float(sub.loc["AAP", "test_tpr_at_fpr1"] - sub.loc["RANDOM", "test_tpr_at_fpr1"]),
                    "pr_gain_aap_minus_random": float(sub.loc["AAP", "test_pr_auc"] - sub.loc["RANDOM", "test_pr_auc"]),
                }
            )
    aap_gain_df = pd.DataFrame(aap_gain_rows)
    aap_gain_df.to_csv(cfg.step2_results_path / "aap_gain_by_fusion.csv", index=False)

    with (cfg.step2_results_path / "step2_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    return res_df
