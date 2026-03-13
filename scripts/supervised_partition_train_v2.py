from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data_processed/cicids2017_all.parquet"
MODEL_DIR = "models"
ARTIFACT_DIR = "artifacts_supervised_partition_v2"
RANDOM_SEED = 42
N_AGENTS = 4
N_SHARED = 3
MAX_REBALANCE_ITERS = 10
MIN_PRIVATE_PER_AGENT = 4

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)


def build_lgbm(random_state: int = RANDOM_SEED, small: bool = False) -> LGBMClassifier:
    if small:
        return LGBMClassifier(
            n_estimators=100,
            learning_rate=0.06,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def safe_series(values: np.ndarray, index: List[str]) -> pd.Series:
    s = pd.Series(values, index=index, dtype=float)
    return s.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("y", "day")]


def select_shared_features(tr_df: pd.DataFrame, feature_cols: List[str], n_shared: int = N_SHARED) -> List[str]:
    model = build_lgbm(small=True)
    model.fit(tr_df[feature_cols], tr_df["y"].values)
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return imp.head(n_shared).index.tolist()


def compute_feature_scores(tr_df: pd.DataFrame, va_df: pd.DataFrame, remaining_features: List[str]) -> pd.DataFrame:
    y_tr = tr_df["y"].values
    y_va = va_df["y"].values

    global_model = build_lgbm(small=False)
    global_model.fit(tr_df[remaining_features], y_tr)
    gbm_imp = safe_series(global_model.feature_importances_, remaining_features)
    gbm_imp = gbm_imp / (gbm_imp.max() + 1e-9)

    mi = safe_series(
        mutual_info_classif(tr_df[remaining_features], y_tr, discrete_features=False, random_state=RANDOM_SEED),
        remaining_features,
    )
    mi = mi / (mi.max() + 1e-9)

    ap_scores = []
    auc_scores = []
    for col in remaining_features:
        model = build_lgbm(random_state=RANDOM_SEED, small=True)
        model.fit(tr_df[[col]], y_tr)
        p_va = model.predict_proba(va_df[[col]])[:, 1]
        ap_scores.append(average_precision_score(y_va, p_va))
        auc_scores.append(roc_auc_score(y_va, p_va))

    ap = safe_series(np.asarray(ap_scores), remaining_features)
    auc = safe_series(np.asarray(auc_scores), remaining_features)
    ap = ap / (ap.max() + 1e-9)
    auc = auc / (auc.max() + 1e-9)

    score = 0.40 * gbm_imp + 0.20 * mi + 0.30 * ap + 0.10 * auc
    df = pd.DataFrame(
        {
            "feature": remaining_features,
            "gbm_importance": gbm_imp.values,
            "mutual_info": mi.values,
            "single_ap": ap.values,
            "single_auc": auc.values,
            "supervised_score": score.values,
        }
    ).sort_values("supervised_score", ascending=False, ignore_index=True)
    return df


def initial_balanced_partition(
    tr_df: pd.DataFrame,
    remaining_features: List[str],
    feature_score_df: pd.DataFrame,
    n_agents: int = N_AGENTS,
) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {f"Agent_{i}": [] for i in range(n_agents)}
    score_map = dict(zip(feature_score_df["feature"], feature_score_df["supervised_score"]))
    corr = tr_df[remaining_features].corr().abs().fillna(0.0)
    ordered = feature_score_df["feature"].tolist()

    # Snake-draft the strongest features so one agent cannot absorb all high-value columns.
    snake = list(range(n_agents)) + list(range(n_agents - 1, -1, -1))
    for idx, feat in enumerate(ordered[: 2 * n_agents]):
        g = f"Agent_{snake[idx % len(snake)]}"
        groups[g].append(feat)

    assigned = set(sum(groups.values(), []))
    target_size = len(remaining_features) / n_agents
    total_score = float(feature_score_df["supervised_score"].sum())
    target_score = total_score / n_agents

    for feat in ordered:
        if feat in assigned:
            continue
        best_group = None
        best_obj = None
        feat_score = score_map[feat]
        for g_name, g_feats in groups.items():
            redundancy = float(corr.loc[feat, g_feats].mean()) if g_feats else 0.0
            cur_score = sum(score_map[x] for x in g_feats)
            score_after = cur_score + feat_score
            size_after = len(g_feats) + 1
            all_scores = []
            for h_name, h_feats in groups.items():
                s = sum(score_map[x] for x in h_feats)
                if h_name == g_name:
                    s += feat_score
                all_scores.append(s)
            dominance = max(all_scores) - min(all_scores)
            obj = (
                0.40 * redundancy
                + 0.25 * abs(score_after - target_score)
                + 0.15 * abs(size_after - target_size)
                + 0.20 * dominance
            )
            if best_obj is None or obj < best_obj:
                best_obj = obj
                best_group = g_name
        groups[best_group].append(feat)
        assigned.add(feat)
    return groups


def train_agents(tr_df: pd.DataFrame, groups: Dict[str, List[str]], shared_features: List[str], small: bool = False) -> Dict[str, Dict]:
    agents: Dict[str, Dict] = {}
    for agent_name, private_cols in groups.items():
        cols = shared_features + private_cols
        model = build_lgbm(random_state=RANDOM_SEED, small=small)
        model.fit(tr_df[cols], tr_df["y"].values)
        agents[agent_name] = {
            "model": model,
            "cols": cols,
            "shared_cols": list(shared_features),
            "private_cols": list(private_cols),
        }
    return agents


def evaluate_agents(agents: Dict[str, Dict], va_df: pd.DataFrame) -> pd.DataFrame:
    y_va = va_df["y"].values
    rows = []
    for agent_name, bundle in agents.items():
        p = bundle["model"].predict_proba(va_df[bundle["cols"]])[:, 1]
        rows.append(
            {
                "agent": agent_name,
                "n_features": len(bundle["cols"]),
                "n_private": len(bundle["private_cols"]),
                "val_pr_auc": float(average_precision_score(y_va, p)),
                "val_roc_auc": float(roc_auc_score(y_va, p)),
            }
        )
    return pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False, ignore_index=True)


def top2_proxy_score(eval_df: pd.DataFrame) -> float:
    vals = eval_df.sort_values("val_pr_auc", ascending=False)["val_pr_auc"].tolist()
    if len(vals) < 2:
        return vals[0] if vals else 0.0
    return float(np.mean(vals[:2]))


def move_candidates(groups: Dict[str, List[str]], score_map: Dict[str, float], donor: str, max_candidates: int = 6) -> List[str]:
    feats = groups[donor]
    feats = sorted(feats, key=lambda x: score_map.get(x, 0.0), reverse=True)
    if len(feats) <= MIN_PRIVATE_PER_AGENT:
        return []
    return feats[:max_candidates]


def rebalance_for_top2(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    shared_features: List[str],
    feature_score_df: pd.DataFrame,
    max_iters: int = MAX_REBALANCE_ITERS,
) -> Tuple[Dict[str, List[str]], pd.DataFrame, List[Dict[str, float]]]:
    score_map = dict(zip(feature_score_df["feature"], feature_score_df["supervised_score"]))
    history: List[Dict[str, float]] = []

    current_groups = deepcopy(groups)
    current_agents = train_agents(tr_df, current_groups, shared_features, small=True)
    current_eval = evaluate_agents(current_agents, va_df)
    current_proxy = top2_proxy_score(current_eval)

    for it in range(max_iters):
        sorted_eval = current_eval.sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
        donor = str(sorted_eval.loc[0, "agent"])
        receiver = str(sorted_eval.loc[len(sorted_eval) - 1, "agent"])
        donor_pr = float(sorted_eval.loc[0, "val_pr_auc"])
        receiver_pr = float(sorted_eval.loc[len(sorted_eval) - 1, "val_pr_auc"])
        gap = donor_pr - receiver_pr
        if gap < 0.05:
            break

        best_move = None
        best_groups = None
        best_eval = None
        best_proxy = current_proxy

        for feat in move_candidates(current_groups, score_map, donor):
            if feat in current_groups[receiver]:
                continue
            cand_groups = deepcopy(current_groups)
            cand_groups[donor].remove(feat)
            cand_groups[receiver].append(feat)

            cand_agents = train_agents(tr_df, cand_groups, shared_features, small=True)
            cand_eval = evaluate_agents(cand_agents, va_df)
            cand_proxy = top2_proxy_score(cand_eval)

            cand_sorted = cand_eval.sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
            cand_gap = float(cand_sorted.loc[0, "val_pr_auc"] - cand_sorted.loc[len(cand_sorted) - 1, "val_pr_auc"])
            donor_after = float(cand_eval.set_index("agent").loc[donor, "val_pr_auc"])
            receiver_after = float(cand_eval.set_index("agent").loc[receiver, "val_pr_auc"])

            # Accept if top2 proxy improves, or if balance improves substantially with minimal top2 damage.
            improved = cand_proxy > best_proxy + 1e-4
            balance_trade = (cand_gap < gap - 0.03) and (cand_proxy >= current_proxy - 0.005) and (receiver_after > receiver_pr + 0.02)
            donor_not_collapsed = donor_after >= donor_pr - 0.05
            if donor_not_collapsed and (improved or balance_trade):
                best_move = feat
                best_groups = cand_groups
                best_eval = cand_eval
                best_proxy = cand_proxy

        if best_move is None:
            break

        history.append(
            {
                "iter": float(it + 1),
                "moved_feature": best_move,
                "from_agent": donor,
                "to_agent": receiver,
                "top2_proxy_before": float(current_proxy),
                "top2_proxy_after": float(best_proxy),
            }
        )
        current_groups = best_groups
        current_eval = best_eval
        current_proxy = best_proxy

    final_agents = train_agents(tr_df, current_groups, shared_features, small=False)
    final_eval = evaluate_agents(final_agents, va_df)
    return current_groups, final_eval, history


def export_partition(
    groups: Dict[str, List[str]],
    feature_score_df: pd.DataFrame,
    shared_features: List[str],
    eval_df: pd.DataFrame,
    history: List[Dict[str, float]],
) -> None:
    score_map = feature_score_df.set_index("feature").to_dict(orient="index")
    partition_json = {"shared_features": shared_features, "groups": {}, "rebalance_history": history}
    group_rows = []
    for agent_name, feats in groups.items():
        partition_json["groups"][agent_name] = [
            {"feature": f, **{k: float(v) for k, v in score_map[f].items()}} for f in feats
        ]
        total_score = feature_score_df.set_index("feature").loc[feats, "supervised_score"].sum() if feats else 0.0
        group_rows.append(
            {
                "agent": agent_name,
                "n_private": len(feats),
                "private_features": ", ".join(feats),
                "group_supervised_score_sum": float(total_score),
            }
        )

    with open(Path(ARTIFACT_DIR) / "supervised_partition_v2.json", "w", encoding="utf-8") as f:
        json.dump(partition_json, f, ensure_ascii=False, indent=2)

    pd.DataFrame(group_rows).to_csv(Path(ARTIFACT_DIR) / "supervised_partition_groups_v2.csv", index=False, encoding="utf-8-sig")
    feature_score_df.to_csv(Path(ARTIFACT_DIR) / "supervised_feature_scores_v2.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(Path(ARTIFACT_DIR) / "supervised_agent_validation_v2.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(Path(ARTIFACT_DIR) / "supervised_rebalance_history_v2.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    train_days = {"tuesday", "wednesday", "thursday"}
    train_df = df[df["day"].isin(train_days)].copy()
    tr_df, va_df = train_test_split(train_df, test_size=0.2, stratify=train_df["y"], random_state=RANDOM_SEED)

    feature_cols = get_feature_columns(df)
    shared_features = select_shared_features(tr_df, feature_cols, n_shared=N_SHARED)
    remaining_features = [c for c in feature_cols if c not in shared_features]

    print("[INFO] shared features:", shared_features)
    print(f"[INFO] remaining features: {len(remaining_features)}")

    feature_score_df = compute_feature_scores(tr_df, va_df, remaining_features)
    init_groups = initial_balanced_partition(tr_df, remaining_features, feature_score_df, n_agents=N_AGENTS)
    init_agents = train_agents(tr_df, init_groups, shared_features, small=True)
    init_eval = evaluate_agents(init_agents, va_df)

    final_groups, final_eval, history = rebalance_for_top2(
        tr_df=tr_df,
        va_df=va_df,
        groups=init_groups,
        shared_features=shared_features,
        feature_score_df=feature_score_df,
        max_iters=MAX_REBALANCE_ITERS,
    )
    final_agents = train_agents(tr_df, final_groups, shared_features, small=False)

    joblib.dump(final_agents, Path(MODEL_DIR) / "local_agents_supervised_balanced.joblib")
    joblib.dump(final_agents, Path(MODEL_DIR) / "local_agents_supervised.joblib")
    joblib.dump(final_agents, Path(MODEL_DIR) / "step2_agents_supervised.joblib")

    export_partition(final_groups, feature_score_df, shared_features, final_eval, history)

    print("\n[INFO] initial validation performance")
    print(init_eval.to_string(index=False))
    print("\n[INFO] final validation performance")
    print(final_eval.to_string(index=False))
    print(f"\n[INFO] top2 proxy before: {top2_proxy_score(init_eval):.4f}")
    print(f"[INFO] top2 proxy after : {top2_proxy_score(final_eval):.4f}")
    print(f"[INFO] rebalance moves   : {len(history)}")
    print("[INFO] saved model      :", Path(MODEL_DIR) / "local_agents_supervised_balanced.joblib")
    print("[INFO] compatibility    :", Path(MODEL_DIR) / "local_agents_supervised.joblib")
    print("[INFO] artifacts        :", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
