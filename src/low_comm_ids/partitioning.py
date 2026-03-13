from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.cluster import AgglomerativeClustering

from .config import ExperimentConfig


GroupDict = Dict[str, List[str]]


def select_shared_features(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: ExperimentConfig,
) -> List[str]:
    if not feature_cols:
        return []
    model = LGBMClassifier(**cfg.global_lgbm_params)
    model.fit(train_df[list(feature_cols)], train_df["y"].values)
    top_idx = np.argsort(model.feature_importances_)[::-1][: cfg.n_shared_features]
    return [list(feature_cols)[i] for i in top_idx]


def partition_random(features: Sequence[str], n_agents: int, seed: int) -> GroupDict:
    feats = list(features)
    rng = np.random.default_rng(seed)
    rng.shuffle(feats)
    chunks = np.array_split(feats, n_agents)
    return {f"Agent_{i}": list(chunk) for i, chunk in enumerate(chunks)}


def _fit_agglomerative(dist_matrix: np.ndarray, n_agents: int) -> np.ndarray:
    kwargs = {
        "n_clusters": n_agents,
        "linkage": "complete",
    }
    try:
        model = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", **kwargs)
    return model.fit_predict(dist_matrix)


def partition_aap(
    train_df: pd.DataFrame,
    features: Sequence[str],
    n_agents: int,
) -> GroupDict:
    features = list(features)
    if not features:
        return {f"Agent_{i}": [] for i in range(n_agents)}

    x_train = train_df[features]
    std_vals = x_train.std()
    valid_features = std_vals[std_vals > 0].index.tolist()
    constant_features = [f for f in features if f not in valid_features]

    groups = {f"Agent_{i}": [] for i in range(n_agents)}
    if valid_features:
        corr_matrix = x_train[valid_features].corr().abs().fillna(0.0)
        dist_matrix = 1.0 - corr_matrix.to_numpy()
        labels = _fit_agglomerative(dist_matrix, n_agents=n_agents)
        for feature_name, cluster_id in zip(valid_features, labels):
            groups[f"Agent_{cluster_id}"] .append(feature_name)

    for feature_name in constant_features:
        smallest = min(groups, key=lambda key: len(groups[key]))
        groups[smallest].append(feature_name)

    return groups


def save_partition_description(
    path: str | Path,
    method_name: str,
    shared_features: Sequence[str],
    groups: GroupDict,
) -> None:
    payload = {
        "method": method_name,
        "shared_features": list(shared_features),
        "groups": groups,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
