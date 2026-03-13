from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from .config import ExperimentConfig
from .data import get_feature_cols, load_processed_dataset, make_standard_splits
from .fusion import l1_vote_fraction, make_l2_outputs
from .metrics import build_best_l0_row, evaluate_method
from .models import (
    fit_l3_pca_bundle,
    predict_agent_score_matrix,
    predict_l3_pca_scores,
    train_agent_models,
    train_centralized_model,
)
from .partitioning import partition_random, save_partition_description, select_shared_features


def run_step1_baselines(cfg: ExperimentConfig) -> pd.DataFrame:
    cfg.ensure_dirs()
    df = load_processed_dataset(cfg)
    tr_df, va_df, test_df, calib_threshold_df, calib_report_df = make_standard_splits(df, cfg)
    feature_cols = get_feature_cols(df)

    shared_features = select_shared_features(tr_df, feature_cols, cfg)
    private_features = [c for c in feature_cols if c not in shared_features]
    random_groups = partition_random(private_features, cfg.n_agents, cfg.random_state)

    save_partition_description(
        cfg.step1_models_path / "step1_random_partition.json",
        method_name="RANDOM_BASELINE",
        shared_features=shared_features,
        groups=random_groups,
    )

    agents = train_agent_models(tr_df, random_groups, shared_features, cfg)
    joblib.dump(agents, cfg.step1_models_path / "step1_random_agents.joblib")

    agent_names, va_mat = predict_agent_score_matrix(agents, va_df)
    _, te_mat = predict_agent_score_matrix(agents, test_df)
    _, ca_thr_mat = predict_agent_score_matrix(agents, calib_threshold_df)
    _, ca_rep_mat = predict_agent_score_matrix(agents, calib_report_df)

    rows: List[Dict[str, object]] = []
    for idx, agent_name in enumerate(agent_names):
        rows.append(
            evaluate_method(
                method=f"L0_{agent_name}",
                level="L0",
                partition_mode="RANDOM",
                comm_cost_units=1.0,
                test_scores=te_mat[:, idx],
                test_y=test_df["y"].values,
                calib_threshold_scores=ca_thr_mat[:, idx],
                calib_threshold_y=calib_threshold_df["y"].values,
                calib_report_scores=ca_rep_mat[:, idx],
                calib_report_y=calib_report_df["y"].values,
                target_fpr=cfg.target_fpr,
            )
        )

    rows.append(
        evaluate_method(
            method="L1_vote",
            level="L1",
            partition_mode="RANDOM",
            comm_cost_units=len(agent_names) / 32.0,
            test_scores=l1_vote_fraction(te_mat, cfg.l1_threshold),
            test_y=test_df["y"].values,
            calib_threshold_scores=l1_vote_fraction(ca_thr_mat, cfg.l1_threshold),
            calib_threshold_y=calib_threshold_df["y"].values,
            calib_report_scores=l1_vote_fraction(ca_rep_mat, cfg.l1_threshold),
            calib_report_y=calib_report_df["y"].values,
            target_fpr=cfg.target_fpr,
        )
    )

    l2_outputs = make_l2_outputs(
        va_mat=va_mat,
        te_mat=te_mat,
        ca_thr_mat=ca_thr_mat,
        ca_rep_mat=ca_rep_mat,
        y_va=va_df["y"].values,
        cfg=cfg,
    )
    for method_name, payload in l2_outputs.items():
        if method_name in {"L2_attention_ap", "L2_attention_robust"}:
            continue
        rows.append(
            evaluate_method(
                method=method_name,
                level="L2",
                partition_mode="RANDOM",
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

    for dim in cfg.l3_dims:
        bundle = fit_l3_pca_bundle(tr_df, random_groups, shared_features, dim, cfg)
        joblib.dump(bundle, cfg.step1_models_path / f"step1_l3_pca_d{dim}.joblib")
        rows.append(
            evaluate_method(
                method=f"L3_pca_d{dim}",
                level="L3",
                partition_mode="RANDOM",
                comm_cost_units=float(bundle["comm_cost"]),
                test_scores=predict_l3_pca_scores(bundle, test_df),
                test_y=test_df["y"].values,
                calib_threshold_scores=predict_l3_pca_scores(bundle, calib_threshold_df),
                calib_threshold_y=calib_threshold_df["y"].values,
                calib_report_scores=predict_l3_pca_scores(bundle, calib_report_df),
                calib_report_y=calib_report_df["y"].values,
                target_fpr=cfg.target_fpr,
            )
        )

    centralized = train_centralized_model(tr_df, feature_cols, cfg)
    joblib.dump(centralized, cfg.step1_models_path / "step1_centralized.joblib")
    rows.append(
        evaluate_method(
            method="Centralized",
            level="Centralized",
            partition_mode="CENTRALIZED",
            comm_cost_units=float(len(feature_cols)),
            test_scores=centralized.predict_proba(test_df[feature_cols])[:, 1],
            test_y=test_df["y"].values,
            calib_threshold_scores=centralized.predict_proba(calib_threshold_df[feature_cols])[:, 1],
            calib_threshold_y=calib_threshold_df["y"].values,
            calib_report_scores=centralized.predict_proba(calib_report_df[feature_cols])[:, 1],
            calib_report_y=calib_report_df["y"].values,
            target_fpr=cfg.target_fpr,
        )
    )

    res_df = pd.DataFrame(rows).sort_values(["comm_cost_units", "method"]).reset_index(drop=True)
    res_df.to_csv(cfg.step1_results_path / "baseline_metrics.csv", index=False)

    plot_methods = [
        "L1_vote",
        "L2_mean",
        "L2_trimmed_mean",
        "L2_stacking_lr",
        "Centralized",
    ] + [f"L3_pca_d{d}" for d in cfg.l3_dims]
    plot_df = pd.concat(
        [build_best_l0_row(res_df), res_df[res_df["method"].isin(plot_methods)]],
        ignore_index=True,
    )
    plot_df = plot_df.sort_values(["comm_cost_units", "method"]).reset_index(drop=True)
    plot_df.to_csv(cfg.step1_results_path / "baseline_plot_table.csv", index=False)

    with (cfg.step1_results_path / "step1_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    return res_df
