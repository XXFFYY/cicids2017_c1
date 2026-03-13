#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal robustness experiment for the current thesis storyline.

Goal:
- Compare AAP vs supervised
- Compare Global_Best_L0 vs Global_Best_Pair_L2
- Under two small robustness families:
  1) calibration split changes
  2) local agent perturbations

Outputs:
- results_robustness_minimal/robustness_results.csv
- results_robustness_minimal/robustness_summary.csv
- results_robustness_minimal/robustness_summary.md
- plots (*.png)
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path('.')
DATA_PATH = ROOT / 'data_processed' / 'cicids2017_all.parquet'
MODEL_PATHS = {
    'aap': ROOT / 'models' /'step2'/ 'step2_agents_aap.joblib',
    'supervised': ROOT / 'models' / 'local_agents_supervised.joblib',
}
OUTDIR = ROOT / 'results_robustness_minimal'
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_FPR = 0.01


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.sum(y_true) == 0:
        return float('nan')
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def threshold_from_negatives(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = TARGET_FPR) -> float:
    neg = y_score[y_true == 0]
    if len(neg) == 0:
        return 1.0
    return float(np.quantile(neg, 1.0 - target_fpr))


def tpr_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    pos = y_score[y_true == 1]
    if len(pos) == 0:
        return float('nan')
    return float(np.mean(pos >= thr))


def fpr_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    neg = y_score[y_true == 0]
    if len(neg) == 0:
        return float('nan')
    return float(np.mean(neg >= thr))


def normalize_cols(cols) -> List[str]:
    if isinstance(cols, str):
        out = [cols]
    elif isinstance(cols, pd.Index):
        out = cols.astype(str).tolist()
    elif isinstance(cols, np.ndarray):
        out = [str(x) for x in np.atleast_1d(cols).tolist()]
    elif isinstance(cols, (list, tuple, set)):
        out = [str(x) for x in cols]
    else:
        out = [str(cols)]
    out = [c for c in out if c]
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def to_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    X = df.loc[:, cols].copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = X.to_numpy(dtype=np.float32, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def split_data(df: pd.DataFrame):
    df = df.copy()
    df['day'] = df['day'].astype(str).str.strip().str.lower()
    train_days = {'tuesday', 'wednesday', 'thursday'}
    train_df = df[df['day'].isin(train_days)].copy()
    friday_df = df[df['day'] == 'friday'].copy().reset_index(drop=True)
    monday_df = df[df['day'] == 'monday'].copy().reset_index(drop=True)
    if len(train_df) == 0 or len(friday_df) == 0 or len(monday_df) == 0:
        raise ValueError(f'Bad day split: train={train_df.shape}, friday={friday_df.shape}, monday={monday_df.shape}')
    _, va_df = train_test_split(train_df, test_size=0.2, stratify=train_df['y'], random_state=SEED)
    ca_df, cb_df = train_test_split(monday_df, test_size=0.5, stratify=monday_df['y'], random_state=SEED)
    fr_a, fr_b = train_test_split(friday_df, test_size=0.5, stratify=friday_df['y'], random_state=SEED)
    return va_df.reset_index(drop=True), ca_df.reset_index(drop=True), cb_df.reset_index(drop=True), friday_df, fr_a.reset_index(drop=True), fr_b.reset_index(drop=True)


def load_agents(path: Path) -> Dict[str, dict]:
    obj = joblib.load(path)
    agents = obj['agents'] if isinstance(obj, dict) and 'agents' in obj else obj
    if not isinstance(agents, dict):
        raise ValueError(f'Unexpected agent bundle format: {path}')
    return agents


def predict_all(agents: Dict[str, dict], df_map: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for agent_name, bundle in agents.items():
        cols = [c for c in normalize_cols(bundle.get('cols', [])) if all(c in df_map[k].columns for k in df_map)]
        if not cols:
            continue
        model = bundle['model']
        pred = {'cols': np.array(cols, dtype=object)}
        for split_name, df in df_map.items():
            pred[split_name] = model.predict_proba(to_matrix(df, cols))[:, 1]
        out[agent_name] = pred
    if len(out) < 2:
        raise ValueError('Fewer than 2 valid agents found.')
    return out


def choose_best_single(preds: Dict[str, Dict[str, np.ndarray]], y_va: np.ndarray) -> str:
    ranked = []
    for name, bundle in preds.items():
        ranked.append((name, safe_pr_auc(y_va, bundle['va']), safe_roc_auc(y_va, bundle['va'])))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return ranked[0][0]


def pair_weight(p_i: np.ndarray, p_j: np.ndarray, y_va: np.ndarray) -> np.ndarray:
    aps = np.array([safe_pr_auc(y_va, p_i), safe_pr_auc(y_va, p_j)], dtype=float)
    aps = np.nan_to_num(aps, nan=0.0)
    w = np.exp(aps / 0.1)
    w = w / np.clip(w.sum(), 1e-12, None)
    return w


def fuse_pair_mean(p_i: np.ndarray, p_j: np.ndarray) -> np.ndarray:
    return 0.5 * (p_i + p_j)


def fuse_pair_weighted(p_i: np.ndarray, p_j: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w[0] * p_i + w[1] * p_j


def choose_best_pair(preds: Dict[str, Dict[str, np.ndarray]], y_va: np.ndarray, method: str = 'weighted') -> Tuple[Tuple[str, str], np.ndarray]:
    best = None
    best_key = None
    names = sorted(preds.keys())
    for a, b in combinations(names, 2):
        p_a = preds[a]['va']
        p_b = preds[b]['va']
        if method == 'weighted':
            w = pair_weight(p_a, p_b, y_va)
            p = fuse_pair_weighted(p_a, p_b, w)
        else:
            w = np.array([0.5, 0.5], dtype=float)
            p = fuse_pair_mean(p_a, p_b)
        key = (safe_pr_auc(y_va, p), safe_roc_auc(y_va, p))
        if best_key is None or key > best_key:
            best_key = key
            best = (a, b)
            best_w = w
    assert best is not None
    return best, best_w


def apply_feature_dropout(df: pd.DataFrame, cols: List[str], drop_ratio: float) -> pd.DataFrame:
    if drop_ratio <= 0:
        return df
    cols = list(cols)
    rng = np.random.default_rng(SEED + int(drop_ratio * 1000) + len(cols))
    k = max(1, int(round(len(cols) * drop_ratio)))
    chosen = rng.choice(cols, size=min(k, len(cols)), replace=False)
    out = df.copy()
    out.loc[:, list(chosen)] = 0.0
    return out


def apply_score_noise(p: np.ndarray, sigma: float) -> np.ndarray:
    rng = np.random.default_rng(SEED + int(sigma * 1000))
    noisy = p + rng.normal(0.0, sigma, size=len(p))
    return np.clip(noisy, 0.0, 1.0)


def evaluate_single(y_va: np.ndarray, y_te: np.ndarray, y_cb: np.ndarray,
                    p_va: np.ndarray, p_te: np.ndarray, p_cb: np.ndarray) -> Dict[str, float]:
    thr = threshold_from_negatives(y_cb, p_cb, TARGET_FPR)
    return {
        'val_pr_auc': safe_pr_auc(y_va, p_va),
        'val_roc_auc': safe_roc_auc(y_va, p_va),
        'pr_auc': safe_pr_auc(y_te, p_te),
        'roc_auc': safe_roc_auc(y_te, p_te),
        'tpr_at_fpr1': tpr_at_threshold(y_te, p_te, thr),
        'monitor_fpr': fpr_at_threshold(y_cb, p_cb, thr),
        'decision_threshold': thr,
    }


def evaluate_pair(y_va: np.ndarray, y_te: np.ndarray, y_cb: np.ndarray,
                  pa_va: np.ndarray, pa_te: np.ndarray, pa_cb: np.ndarray,
                  pb_va: np.ndarray, pb_te: np.ndarray, pb_cb: np.ndarray,
                  method: str, w: np.ndarray | None = None) -> Dict[str, float]:
    if method == 'weighted':
        ww = pair_weight(pa_va, pb_va, y_va) if w is None else w
        p_va = fuse_pair_weighted(pa_va, pb_va, ww)
        p_te = fuse_pair_weighted(pa_te, pb_te, ww)
        p_cb = fuse_pair_weighted(pa_cb, pb_cb, ww)
    else:
        ww = np.array([0.5, 0.5], dtype=float)
        p_va = fuse_pair_mean(pa_va, pb_va)
        p_te = fuse_pair_mean(pa_te, pb_te)
        p_cb = fuse_pair_mean(pa_cb, pb_cb)
    out = evaluate_single(y_va, y_te, y_cb, p_va, p_te, p_cb)
    out['w0'] = float(ww[0])
    out['w1'] = float(ww[1])
    return out


def run_mode(mode: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    va_df, ca_df, cb_df, te_df, fr_a_df, fr_b_df = split_data(df)
    agents = load_agents(MODEL_PATHS[mode])
    base_preds = predict_all(agents, {
        'va': va_df,
        'ca': ca_df,
        'cb': cb_df,
        'te': te_df,
        'fr_a': fr_a_df,
        'fr_b': fr_b_df,
    })
    y_va = va_df['y'].to_numpy().astype(int)
    y_ca = ca_df['y'].to_numpy().astype(int)
    y_cb = cb_df['y'].to_numpy().astype(int)
    y_te = te_df['y'].to_numpy().astype(int)
    y_fr_a = fr_a_df['y'].to_numpy().astype(int)
    y_fr_b = fr_b_df['y'].to_numpy().astype(int)

    single = choose_best_single(base_preds, y_va)
    pair, pair_w = choose_best_pair(base_preds, y_va, method='weighted')

    rows = []

    # Base scenarios
    base_single = evaluate_single(y_va, y_te, y_cb, base_preds[single]['va'], base_preds[single]['te'], base_preds[single]['cb'])
    rows.append({'mode': mode, 'object': 'Global_Best_L0', 'scenario_family': 'base', 'scenario': 'base', 'agent': single, 'pair': '', 'comm_cost': 1, **base_single})

    base_pair = evaluate_pair(
        y_va, y_te, y_cb,
        base_preds[pair[0]]['va'], base_preds[pair[0]]['te'], base_preds[pair[0]]['cb'],
        base_preds[pair[1]]['va'], base_preds[pair[1]]['te'], base_preds[pair[1]]['cb'],
        method='weighted', w=pair_w,
    )
    rows.append({'mode': mode, 'object': 'Global_Best_Pair_L2', 'scenario_family': 'base', 'scenario': 'base', 'agent': '', 'pair': f'{pair[0]}+{pair[1]}', 'comm_cost': 2, **base_pair})

    # Calibration drift family
    # swap monday halves
    swap_single = evaluate_single(y_va, y_te, y_ca, base_preds[single]['va'], base_preds[single]['te'], base_preds[single]['ca'])
    rows.append({'mode': mode, 'object': 'Global_Best_L0', 'scenario_family': 'calibration', 'scenario': 'monday_swap', 'agent': single, 'pair': '', 'comm_cost': 1, **swap_single})

    swap_pair = evaluate_pair(
        y_va, y_te, y_ca,
        base_preds[pair[0]]['va'], base_preds[pair[0]]['te'], base_preds[pair[0]]['ca'],
        base_preds[pair[1]]['va'], base_preds[pair[1]]['te'], base_preds[pair[1]]['ca'],
        method='weighted', w=pair_w,
    )
    rows.append({'mode': mode, 'object': 'Global_Best_Pair_L2', 'scenario_family': 'calibration', 'scenario': 'monday_swap', 'agent': '', 'pair': f'{pair[0]}+{pair[1]}', 'comm_cost': 2, **swap_pair})

    # friday half calibration sanity check
    fr_single = evaluate_single(y_va, y_fr_b, y_fr_a, base_preds[single]['va'], base_preds[single]['fr_b'], base_preds[single]['fr_a'])
    rows.append({'mode': mode, 'object': 'Global_Best_L0', 'scenario_family': 'calibration', 'scenario': 'friday_halfswap', 'agent': single, 'pair': '', 'comm_cost': 1, **fr_single})

    fr_pair = evaluate_pair(
        y_va, y_fr_b, y_fr_a,
        base_preds[pair[0]]['va'], base_preds[pair[0]]['fr_b'], base_preds[pair[0]]['fr_a'],
        base_preds[pair[1]]['va'], base_preds[pair[1]]['fr_b'], base_preds[pair[1]]['fr_a'],
        method='weighted', w=pair_w,
    )
    rows.append({'mode': mode, 'object': 'Global_Best_Pair_L2', 'scenario_family': 'calibration', 'scenario': 'friday_halfswap', 'agent': '', 'pair': f'{pair[0]}+{pair[1]}', 'comm_cost': 2, **fr_pair})

    # Perturbation family
    perturb_specs = [
        ('feature_dropout_30', 0.30, None),
        ('feature_dropout_50', 0.50, None),
        ('score_noise_003', None, 0.03),
    ]

    # Single perturbation: hit selected single
    single_cols = list(base_preds[single]['cols'])
    for scenario, drop_ratio, sigma in perturb_specs:
        if drop_ratio is not None:
            pred = predict_all(agents, {
                'va': apply_feature_dropout(va_df, single_cols, drop_ratio),
                'ca': apply_feature_dropout(ca_df, single_cols, drop_ratio),
                'cb': apply_feature_dropout(cb_df, single_cols, drop_ratio),
                'te': apply_feature_dropout(te_df, single_cols, drop_ratio),
                'fr_a': apply_feature_dropout(fr_a_df, single_cols, drop_ratio),
                'fr_b': apply_feature_dropout(fr_b_df, single_cols, drop_ratio),
            })
            p_va = pred[single]['va']; p_te = pred[single]['te']; p_cb = pred[single]['cb']
        else:
            p_va = apply_score_noise(base_preds[single]['va'], sigma)
            p_te = apply_score_noise(base_preds[single]['te'], sigma)
            p_cb = apply_score_noise(base_preds[single]['cb'], sigma)
        out = evaluate_single(y_va, y_te, y_cb, p_va, p_te, p_cb)
        rows.append({'mode': mode, 'object': 'Global_Best_L0', 'scenario_family': 'perturb', 'scenario': scenario, 'agent': single, 'pair': '', 'comm_cost': 1, **out})

    # Pair perturbations: hit each member separately
    for hit in pair:
        hit_cols = list(base_preds[hit]['cols'])
        for scenario, drop_ratio, sigma in perturb_specs:
            pair_name = f'{pair[0]}+{pair[1]}'
            if drop_ratio is not None:
                pred = predict_all(agents, {
                    'va': apply_feature_dropout(va_df, hit_cols, drop_ratio),
                    'ca': apply_feature_dropout(ca_df, hit_cols, drop_ratio),
                    'cb': apply_feature_dropout(cb_df, hit_cols, drop_ratio),
                    'te': apply_feature_dropout(te_df, hit_cols, drop_ratio),
                    'fr_a': apply_feature_dropout(fr_a_df, hit_cols, drop_ratio),
                    'fr_b': apply_feature_dropout(fr_b_df, hit_cols, drop_ratio),
                })
                pa_va, pa_te, pa_cb = pred[pair[0]]['va'], pred[pair[0]]['te'], pred[pair[0]]['cb']
                pb_va, pb_te, pb_cb = pred[pair[1]]['va'], pred[pair[1]]['te'], pred[pair[1]]['cb']
            else:
                pa_va, pa_te, pa_cb = base_preds[pair[0]]['va'], base_preds[pair[0]]['te'], base_preds[pair[0]]['cb']
                pb_va, pb_te, pb_cb = base_preds[pair[1]]['va'], base_preds[pair[1]]['te'], base_preds[pair[1]]['cb']
                if hit == pair[0]:
                    pa_va = apply_score_noise(pa_va, sigma); pa_te = apply_score_noise(pa_te, sigma); pa_cb = apply_score_noise(pa_cb, sigma)
                else:
                    pb_va = apply_score_noise(pb_va, sigma); pb_te = apply_score_noise(pb_te, sigma); pb_cb = apply_score_noise(pb_cb, sigma)
            out = evaluate_pair(y_va, y_te, y_cb, pa_va, pa_te, pa_cb, pb_va, pb_te, pb_cb, method='weighted', w=pair_w)
            rows.append({'mode': mode, 'object': 'Global_Best_Pair_L2', 'scenario_family': 'perturb', 'scenario': f'{scenario}__hit_{hit}', 'agent': hit, 'pair': pair_name, 'comm_cost': 2, **out})

    res = pd.DataFrame(rows)
    base_map = res[res['scenario'] == 'base'].set_index(['mode', 'object'])[['tpr_at_fpr1', 'monitor_fpr']].rename(columns={'tpr_at_fpr1': 'base_tpr', 'monitor_fpr': 'base_monitor_fpr'})
    res = res.join(base_map, on=['mode', 'object'])
    res['delta_tpr'] = res['tpr_at_fpr1'] - res['base_tpr']
    res['delta_monitor_fpr'] = res['monitor_fpr'] - res['base_monitor_fpr']
    return res


def make_plots(df: pd.DataFrame) -> None:
    base = df[df['scenario'] == 'base'].copy()
    labels = [f"{m}\n{o.replace('Global_', '')}" for m, o in zip(base['mode'], base['object'])]

    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, base['tpr_at_fpr1'])
    plt.ylabel('TPR@1%FPR')
    plt.title('Base performance under fair protocol')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(OUTDIR / 'robustness_base_tpr.png', dpi=200)
    plt.close()

    calib = df[(df['scenario_family'] == 'calibration') & (df['scenario'] != 'base')].copy()
    calib_piv = calib.pivot_table(index=['mode', 'object'], columns='scenario', values='delta_tpr', aggfunc='mean').fillna(0.0)
    if len(calib_piv) > 0:
        ax = calib_piv.plot(kind='bar', figsize=(10, 4.8))
        ax.set_ylabel('Delta TPR@1%FPR')
        ax.set_title('Calibration drift robustness')
        ax.tick_params(axis='x', rotation=20)
        plt.tight_layout()
        plt.savefig(OUTDIR / 'robustness_calibration_delta_tpr.png', dpi=200)
        plt.close()

    pert = df[df['scenario_family'] == 'perturb'].copy()
    pert['short_scenario'] = pert['scenario'].str.replace(r'__hit_.*$', '', regex=True)
    pert_piv = pert.pivot_table(index=['mode', 'object'], columns='short_scenario', values='delta_tpr', aggfunc='mean').fillna(0.0)
    if len(pert_piv) > 0:
        ax = pert_piv.plot(kind='bar', figsize=(10, 4.8))
        ax.set_ylabel('Mean Delta TPR@1%FPR')
        ax.set_title('Local perturbation robustness')
        ax.tick_params(axis='x', rotation=20)
        plt.tight_layout()
        plt.savefig(OUTDIR / 'robustness_perturb_delta_tpr.png', dpi=200)
        plt.close()


def main() -> None:
    frames = []
    for mode in ['aap', 'supervised']:
        frames.append(run_mode(mode))
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(OUTDIR / 'robustness_results.csv', index=False)

    summary = all_df.groupby(['mode', 'object', 'scenario_family', 'scenario'], as_index=False).agg({
        'tpr_at_fpr1': 'mean',
        'monitor_fpr': 'mean',
        'delta_tpr': 'mean',
        'delta_monitor_fpr': 'mean',
        'comm_cost': 'first',
    })
    summary.to_csv(OUTDIR / 'robustness_summary.csv', index=False)

    lines = []
    lines.append('# Minimal robustness experiment summary')
    lines.append('')
    for mode in ['aap', 'supervised']:
        lines.append(f'## {mode}')
        sub = summary[(summary['mode'] == mode) & (summary['scenario'] == 'base')]
        for _, r in sub.iterrows():
            lines.append(f"- {r['object']}: TPR@1%FPR={r['tpr_at_fpr1']:.4f}, monitor_fpr={r['monitor_fpr']:.4f}, cost={int(r['comm_cost'])}")
        lines.append('')
    (OUTDIR / 'robustness_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    make_plots(all_df)
    print(f'[OK] wrote results to {OUTDIR}')


if __name__ == '__main__':
    main()
