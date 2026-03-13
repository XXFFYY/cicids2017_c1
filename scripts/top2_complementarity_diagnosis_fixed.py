from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data_processed/cicids2017_all.parquet"
MODEL_DIR = "models"
OUTPUT_DIR = "results_top2_complementarity"
RANDOM_SEED = 42
TARGET_FPR = 0.01
PARTITIONS = ["random", "aap", "supervised"]
EPS = 1e-12


def threshold_from_calib(p_calib: np.ndarray, target_fpr: float = TARGET_FPR) -> float:
    return float(np.quantile(p_calib, 1.0 - target_fpr))


def tpr_at_threshold(y_true: np.ndarray, p_score: np.ndarray, threshold: float) -> float:
    pos = y_true == 1
    if pos.sum() == 0:
        return 0.0
    return float((p_score[pos] >= threshold).mean())


def fpr_at_threshold(y_true: np.ndarray, p_score: np.ndarray, threshold: float) -> float:
    neg = y_true == 0
    if neg.sum() == 0:
        return float("nan")
    return float((p_score[neg] >= threshold).mean())


def load_day_splits(df: pd.DataFrame):
    train_days = {"tuesday", "wednesday", "thursday"}
    train_df = df[df["day"].isin(train_days)].copy()
    test_df = df[df["day"] == "friday"].copy()
    monday_df = df[df["day"] == "monday"].copy()
    tr_df, va_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["y"],
        random_state=RANDOM_SEED,
    )
    cal_a_df, cal_b_df = train_test_split(
        monday_df,
        test_size=0.5,
        stratify=monday_df["y"],
        random_state=RANDOM_SEED,
    )
    return tr_df, va_df, test_df, cal_a_df, cal_b_df


def build_agent_scores(
    agents: Dict,
    va_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cal_a_df: pd.DataFrame,
    cal_b_df: pd.DataFrame,
):
    rows = []
    for name, bundle in agents.items():
        model = bundle["model"]
        cols = bundle["cols"]
        rows.append({
            "agent": name,
            "cols": cols,
            "p_va": model.predict_proba(va_df[cols])[:, 1],
            "p_te": model.predict_proba(test_df[cols])[:, 1],
            "p_ca": model.predict_proba(cal_a_df[cols])[:, 1],
            "p_cb": model.predict_proba(cal_b_df[cols])[:, 1],
        })
    return rows


def select_anchor(rows: List[Dict], y_va: np.ndarray) -> Tuple[str, Dict, pd.DataFrame]:
    eval_rows = []
    best_key = None
    best_name = None
    best_row = None
    for row in rows:
        thr = threshold_from_calib(row["p_ca"])
        proxy_tpr = tpr_at_threshold(y_va, row["p_va"], thr)
        proxy_pr = average_precision_score(y_va, row["p_va"])
        proxy_roc = roc_auc_score(y_va, row["p_va"])
        key = (proxy_tpr, proxy_pr, proxy_roc)
        eval_rows.append({
            "agent": row["agent"],
            "val_proxy_tpr": proxy_tpr,
            "val_pr_auc": proxy_pr,
            "val_roc_auc": proxy_roc,
            "anchor_threshold_from_cal_a": thr,
        })
        if best_key is None or key > best_key:
            best_key = key
            best_name = row["agent"]
            best_row = row
    return best_name, best_row, pd.DataFrame(eval_rows).sort_values(["val_proxy_tpr", "val_pr_auc", "val_roc_auc"], ascending=False)


def pairwise_complement_rows(rows: List[Dict], anchor_name: str, y_va: np.ndarray, y_test: np.ndarray, y_cb: np.ndarray) -> Tuple[pd.DataFrame, Dict]:
    by_name = {r["agent"]: r for r in rows}
    anchor = by_name[anchor_name]

    # validation AP weights for true Top2 fusion
    agent_names = [r["agent"] for r in rows]
    va_aps = np.array([average_precision_score(y_va, by_name[n]["p_va"]) for n in agent_names])
    base_w = softmax(va_aps / 0.1)
    weight_map = {n: float(w) for n, w in zip(agent_names, base_w)}

    anchor_thr = threshold_from_calib(anchor["p_ca"])
    anchor_pred = anchor["p_te"] >= anchor_thr
    pos_mask = y_test == 1
    pos_total = int(pos_mask.sum())
    anchor_tp = int(np.sum(anchor_pred & pos_mask))
    anchor_tpr = float(anchor_tp / max(pos_total, 1))

    out_rows = []
    best_partner_by_gain = None
    best_gain = -1.0
    for partner_name in agent_names:
        if partner_name == anchor_name:
            continue
        partner = by_name[partner_name]
        partner_thr = threshold_from_calib(partner["p_ca"])
        partner_pred = partner["p_te"] >= partner_thr

        overlap_tp = int(np.sum(anchor_pred & partner_pred & pos_mask))
        partner_tp = int(np.sum(partner_pred & pos_mask))
        union_pred = anchor_pred | partner_pred
        union_tp = int(np.sum(union_pred & pos_mask))
        union_tpr = float(union_tp / max(pos_total, 1))
        added_tp_or = int(np.sum((~anchor_pred) & partner_pred & pos_mask))
        added_tpr_or = float(added_tp_or / max(pos_total, 1))

        idx_a = agent_names.index(anchor_name)
        idx_b = agent_names.index(partner_name)
        pair_w = np.array([base_w[idx_a], base_w[idx_b]], dtype=float)
        pair_w = pair_w / max(pair_w.sum(), EPS)
        pair_te = pair_w[0] * anchor["p_te"] + pair_w[1] * partner["p_te"]
        pair_ca = pair_w[0] * anchor["p_ca"] + pair_w[1] * partner["p_ca"]
        pair_cb = pair_w[0] * anchor["p_cb"] + pair_w[1] * partner["p_cb"]
        pair_thr = threshold_from_calib(pair_ca)
        pair_tpr = tpr_at_threshold(y_test, pair_te, pair_thr)
        pair_monitor_fpr = fpr_at_threshold(y_cb, pair_cb, pair_thr)
        pair_pr_auc = float(average_precision_score(y_test, pair_te))
        pair_roc_auc = float(roc_auc_score(y_test, pair_te))

        # additional positives caught by fused top2 vs anchor-alone thresholding
        pair_pred = pair_te >= pair_thr
        added_tp_pair = int(np.sum((~anchor_pred) & pair_pred & pos_mask))
        added_tpr_pair = float(added_tp_pair / max(pos_total, 1))

        test_corr = float(np.corrcoef(anchor["p_te"], partner["p_te"])[0, 1]) if np.std(anchor["p_te"]) > 0 and np.std(partner["p_te"]) > 0 else float("nan")
        pos_corr = float(np.corrcoef(anchor["p_te"][pos_mask], partner["p_te"][pos_mask])[0, 1]) if pos_mask.sum() > 1 and np.std(anchor["p_te"][pos_mask]) > 0 and np.std(partner["p_te"][pos_mask]) > 0 else float("nan")
        neg_mask = y_test == 0
        neg_corr = float(np.corrcoef(anchor["p_te"][neg_mask], partner["p_te"][neg_mask])[0, 1]) if neg_mask.sum() > 1 and np.std(anchor["p_te"][neg_mask]) > 0 and np.std(partner["p_te"][neg_mask]) > 0 else float("nan")

        row = {
            "anchor_agent": anchor_name,
            "partner_agent": partner_name,
            "anchor_weight": weight_map[anchor_name],
            "partner_weight": weight_map[partner_name],
            "anchor_threshold": anchor_thr,
            "partner_threshold": partner_thr,
            "pair_threshold": pair_thr,
            "anchor_tpr_at_fpr1": anchor_tpr,
            "partner_tpr_at_fpr1": float(partner_tp / max(pos_total, 1)),
            "pair_tpr_at_fpr1": pair_tpr,
            "pair_gain_vs_anchor": pair_tpr - anchor_tpr,
            "pair_pr_auc": pair_pr_auc,
            "pair_roc_auc": pair_roc_auc,
            "pair_monitor_fpr": pair_monitor_fpr,
            "anchor_tp": anchor_tp,
            "partner_tp": partner_tp,
            "overlap_tp": overlap_tp,
            "union_tp_oracle": union_tp,
            "union_tpr_oracle": union_tpr,
            "added_tp_oracle": added_tp_or,
            "added_tpr_oracle": added_tpr_or,
            "added_tp_pair": added_tp_pair,
            "added_tpr_pair": added_tpr_pair,
            "anchor_miss_count": int(np.sum((~anchor_pred) & pos_mask)),
            "test_score_corr": test_corr,
            "test_pos_score_corr": pos_corr,
            "test_neg_score_corr": neg_corr,
            "partner_feature_count": len(partner["cols"]),
            "partner_feature_overlap_with_anchor": len(set(anchor["cols"]) & set(partner["cols"])),
        }
        out_rows.append(row)
        if row["pair_gain_vs_anchor"] > best_gain:
            best_gain = row["pair_gain_vs_anchor"]
            best_partner_by_gain = row

    pair_df = pd.DataFrame(out_rows).sort_values(
        ["pair_gain_vs_anchor", "added_tpr_pair", "added_tpr_oracle", "pair_pr_auc"],
        ascending=False,
    )
    summary = {
        "anchor_agent": anchor_name,
        "anchor_tpr_at_fpr1": anchor_tpr,
        "anchor_tp": anchor_tp,
        "positive_total": pos_total,
        "best_partner_by_pair_gain": best_partner_by_gain,
    }
    return pair_df, summary


def run_partition(mode: str, data_path: str = DATA_PATH, model_dir: str = MODEL_DIR):
    model_path = Path(model_dir) / f"local_agents_{mode}.joblib"
    if not model_path.exists():
        return None, None, {"partition_mode": mode.upper(), "status": "missing_model", "model_path": str(model_path)}

    df = pd.read_parquet(data_path)
    _, va_df, test_df, cal_a_df, cal_b_df = load_day_splits(df)
    agents = joblib.load(model_path)

    y_va = va_df["y"].values
    y_te = test_df["y"].values
    y_cb = cal_b_df["y"].values

    rows = build_agent_scores(agents, va_df, test_df, cal_a_df, cal_b_df)
    anchor_name, anchor_row, anchor_rank_df = select_anchor(rows, y_va)
    pair_df, pair_summary = pairwise_complement_rows(rows, anchor_name, y_va, y_te, y_cb)

    pair_df.insert(0, "partition_mode", mode.upper())
    anchor_rank_df.insert(0, "partition_mode", mode.upper())

    top = pair_df.iloc[0].to_dict() if len(pair_df) else None
    summary = {
        "partition_mode": mode.upper(),
        "status": "ok",
        "anchor_agent": anchor_name,
        "anchor_feature_count": len(anchor_row["cols"]),
        "anchor_validation_ranking": anchor_rank_df.to_dict(orient="records"),
        "best_partner_row": top,
        "best_oracle_partner": pair_df.sort_values(["added_tpr_oracle", "union_tpr_oracle"], ascending=False).iloc[0].to_dict() if len(pair_df) else None,
        "n_agents": len(rows),
    }
    return anchor_rank_df, pair_df, summary


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_pair = []
    all_anchor = []

    for mode in PARTITIONS:
        anchor_df, pair_df, summary = run_partition(mode)
        summaries.append(summary)
        if anchor_df is not None:
            anchor_df.to_csv(out_dir / f"anchor_validation_rank_{mode}.csv", index=False)
            pair_df.to_csv(out_dir / f"top2_complementarity_{mode}.csv", index=False)
            all_pair.append(pair_df)
            all_anchor.append(anchor_df)

    if all_pair:
        pd.concat(all_pair, ignore_index=True).to_csv(out_dir / "top2_complementarity_all.csv", index=False)
    if all_anchor:
        pd.concat(all_anchor, ignore_index=True).to_csv(out_dir / "anchor_validation_rank_all.csv", index=False)

    focus_rows = []
    for s in summaries:
        if s.get("status") != "ok":
            focus_rows.append({"partition_mode": s["partition_mode"], "status": s["status"]})
            continue
        top = s["best_partner_row"] or {}
        focus_rows.append({
            "partition_mode": s["partition_mode"],
            "anchor_agent": s["anchor_agent"],
            "best_partner": top.get("partner_agent"),
            "anchor_tpr_at_fpr1": top.get("anchor_tpr_at_fpr1"),
            "best_pair_tpr_at_fpr1": top.get("pair_tpr_at_fpr1"),
            "pair_gain_vs_anchor": top.get("pair_gain_vs_anchor"),
            "added_tpr_pair": top.get("added_tpr_pair"),
            "added_tpr_oracle": top.get("added_tpr_oracle"),
            "pair_monitor_fpr": top.get("pair_monitor_fpr"),
            "pair_pr_auc": top.get("pair_pr_auc"),
            "status": "ok",
        })
    pd.DataFrame(focus_rows).to_csv(out_dir / "top2_complementarity_summary.csv", index=False)

    with open(out_dir / "top2_complementarity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Top2 互补性诊断摘要",
        "",
        "核心问题：第二个 agent 是否在 Best_L0 之外补抓到新的攻击样本。",
        "",
    ]
    for row in focus_rows:
        mode = row["partition_mode"]
        if row.get("status") != "ok":
            md_lines.append(f"- {mode}: 缺少模型文件，未分析。")
            continue
        md_lines.append(
            f"- {mode}: 锚点 {row['anchor_agent']}，最佳搭档 {row['best_partner']}；"
            f"anchor TPR={row['anchor_tpr_at_fpr1']:.4f}，pair TPR={row['best_pair_tpr_at_fpr1']:.4f}，"
            f"增益={row['pair_gain_vs_anchor']:.4f}，pair monitor FPR={row['pair_monitor_fpr']:.4f}。"
        )
    (out_dir / "top2_complementarity_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
