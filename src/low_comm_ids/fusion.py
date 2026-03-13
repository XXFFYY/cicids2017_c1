from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .config import ExperimentConfig


ScoreSet = Dict[str, Dict[str, np.ndarray]]


def rowwise_trimmed_mean(mat: np.ndarray, trim_k: int = 1) -> np.ndarray:
    if mat.shape[1] <= 2 * trim_k:
        return mat.mean(axis=1)
    sorted_mat = np.sort(mat, axis=1)
    return sorted_mat[:, trim_k:-trim_k].mean(axis=1)


def l1_vote_fraction(mat: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (mat >= threshold).astype(float).mean(axis=1)


def compute_attention_weights(
    validation_mat: np.ndarray,
    y_val: np.ndarray,
    temperature: float,
) -> np.ndarray:
    aps = np.array([
        average_precision_score(y_val, validation_mat[:, i])
        for i in range(validation_mat.shape[1])
    ])
    return softmax(aps / temperature)


def compute_agent_health(
    ref_mat: np.ndarray,
    cur_mat: np.ndarray,
    floor: float = 0.05,
) -> np.ndarray:
    eps = 1e-6
    ref_mean = ref_mat.mean(axis=0)
    cur_mean = cur_mat.mean(axis=0)
    ref_std = ref_mat.std(axis=0) + eps
    cur_std = cur_mat.std(axis=0)

    mean_shift = np.abs(cur_mean - ref_mean) / ref_std
    std_shift = np.abs(cur_std - ref_std) / ref_std
    health = np.exp(-(mean_shift + 0.5 * std_shift))
    return np.clip(health, floor, 1.0)


def make_l2_outputs(
    va_mat: np.ndarray,
    te_mat: np.ndarray,
    ca_thr_mat: np.ndarray,
    ca_rep_mat: np.ndarray,
    y_va: np.ndarray,
    cfg: ExperimentConfig,
) -> ScoreSet:
    outputs: ScoreSet = {}

    outputs["L2_mean"] = {
        "test": te_mat.mean(axis=1),
        "calib_threshold": ca_thr_mat.mean(axis=1),
        "calib_report": ca_rep_mat.mean(axis=1),
    }

    outputs["L2_trimmed_mean"] = {
        "test": rowwise_trimmed_mean(te_mat, trim_k=1),
        "calib_threshold": rowwise_trimmed_mean(ca_thr_mat, trim_k=1),
        "calib_report": rowwise_trimmed_mean(ca_rep_mat, trim_k=1),
    }

    stacker = LogisticRegression(max_iter=1000, random_state=cfg.random_state)
    stacker.fit(va_mat, y_va)
    outputs["L2_stacking_lr"] = {
        "test": stacker.predict_proba(te_mat)[:, 1],
        "calib_threshold": stacker.predict_proba(ca_thr_mat)[:, 1],
        "calib_report": stacker.predict_proba(ca_rep_mat)[:, 1],
    }

    w = compute_attention_weights(va_mat, y_va, cfg.attention_temperature)
    outputs["L2_attention_ap"] = {
        "test": te_mat.dot(w),
        "calib_threshold": ca_thr_mat.dot(w),
        "calib_report": ca_rep_mat.dot(w),
        "weights": w,
    }

    health = compute_agent_health(va_mat, ca_thr_mat, floor=cfg.robust_health_floor)
    w_robust = w * health
    w_robust = w_robust / w_robust.sum()
    outputs["L2_attention_robust"] = {
        "test": te_mat.dot(w_robust),
        "calib_threshold": ca_thr_mat.dot(w_robust),
        "calib_report": ca_rep_mat.dot(w_robust),
        "weights": w_robust,
        "health": health,
    }

    return outputs
