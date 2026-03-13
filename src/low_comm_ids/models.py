from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig


AgentBundle = Dict[str, Dict[str, object]]


def make_lgbm(cfg: ExperimentConfig) -> LGBMClassifier:
    return LGBMClassifier(**cfg.lgbm_params)


def train_agent_models(
    train_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    shared_features: Sequence[str],
    cfg: ExperimentConfig,
) -> AgentBundle:
    models: AgentBundle = {}
    for agent_name, private_cols in groups.items():
        full_cols = list(shared_features) + list(private_cols)
        model = make_lgbm(cfg)
        model.fit(train_df[full_cols], train_df["y"].values)
        models[agent_name] = {
            "model": model,
            "shared_cols": list(shared_features),
            "private_cols": list(private_cols),
            "cols": full_cols,
        }
    return models


def predict_agent_score_matrix(bundle: AgentBundle, df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    names = list(bundle.keys())
    scores = []
    for name in names:
        model = bundle[name]["model"]
        cols = bundle[name]["cols"]
        score = model.predict_proba(df[cols])[:, 1]
        scores.append(score)
    if not scores:
        return names, np.empty((len(df), 0))
    return names, np.column_stack(scores)


class L3PCABundle(dict):
    pass


def fit_l3_pca_bundle(
    train_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    shared_features: Sequence[str],
    pca_dim: int,
    cfg: ExperimentConfig,
) -> L3PCABundle:
    bundle: L3PCABundle = L3PCABundle()
    bundle["agents"] = {}
    bundle["requested_dim"] = pca_dim

    train_embeddings = []
    total_dims = 0

    for agent_name, private_cols in groups.items():
        full_cols = list(shared_features) + list(private_cols)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(train_df[full_cols].to_numpy())
        n_components = max(1, min(pca_dim, x_scaled.shape[0], x_scaled.shape[1]))
        pca = PCA(n_components=n_components, random_state=cfg.random_state)
        z_train = pca.fit_transform(x_scaled)
        total_dims += n_components
        train_embeddings.append(z_train)
        bundle["agents"][agent_name] = {
            "cols": full_cols,
            "scaler": scaler,
            "pca": pca,
            "n_components": n_components,
        }

    x_train = np.concatenate(train_embeddings, axis=1)
    model = make_lgbm(cfg)
    model.fit(x_train, train_df["y"].values)
    bundle["model"] = model
    bundle["comm_cost"] = total_dims
    return bundle


def predict_l3_pca_scores(bundle: L3PCABundle, df: pd.DataFrame) -> np.ndarray:
    embeddings = []
    for agent_name in bundle["agents"]:
        meta = bundle["agents"][agent_name]
        x_scaled = meta["scaler"].transform(df[meta["cols"]].to_numpy())
        z = meta["pca"].transform(x_scaled)
        embeddings.append(z)
    x = np.concatenate(embeddings, axis=1)
    return bundle["model"].predict_proba(x)[:, 1]


def train_centralized_model(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cfg: ExperimentConfig,
) -> LGBMClassifier:
    model = make_lgbm(cfg)
    model.fit(train_df[list(feature_cols)], train_df["y"].values)
    return model
