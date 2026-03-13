from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _negative_scores(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    neg = scores[y_true == 0]
    return neg if len(neg) > 0 else scores


def threshold_at_target_fpr(
    calib_scores: np.ndarray,
    calib_y: np.ndarray,
    target_fpr: float,
) -> float:
    neg_scores = _negative_scores(calib_scores, calib_y)
    return float(np.quantile(neg_scores, 1.0 - target_fpr))


def tpr_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y_pred = (scores >= threshold).astype(int)
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def fpr_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    y_pred = (scores >= threshold).astype(int)
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")


def safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan")


def evaluate_method(
    method: str,
    test_scores: np.ndarray,
    test_y: np.ndarray,
    calib_threshold_scores: np.ndarray,
    calib_threshold_y: np.ndarray,
    calib_report_scores: np.ndarray,
    calib_report_y: np.ndarray,
    target_fpr: float,
    comm_cost_units: float,
    level: str,
    partition_mode: str,
) -> Dict[str, object]:
    threshold = threshold_at_target_fpr(calib_threshold_scores, calib_threshold_y, target_fpr)
    return {
        "method": method,
        "level": level,
        "partition_mode": partition_mode,
        "comm_cost_units": float(comm_cost_units),
        "test_roc_auc": safe_roc_auc(test_y, test_scores),
        "test_pr_auc": safe_pr_auc(test_y, test_scores),
        "test_tpr_at_fpr1": tpr_at_threshold(test_y, test_scores, threshold),
        "monitor_fpr": fpr_at_threshold(calib_report_y, calib_report_scores, threshold),
        "decision_threshold": threshold,
    }


def build_best_l0_row(df: pd.DataFrame) -> pd.DataFrame:
    l0_df = df[df["level"] == "L0"].copy()
    if l0_df.empty:
        return pd.DataFrame(columns=df.columns)
    best = l0_df.sort_values("test_pr_auc", ascending=False).iloc[[0]].copy()
    best.loc[:, "method"] = "Best_L0"
    return best
