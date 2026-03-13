#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-partition fair comparison (fixed).

Fixes compared with the original version:
1. Normalize agent `cols` so a single column string is treated as [col].
2. Detect and report empty / missing feature sets with clear agent+mode context.
3. Skip invalid agents instead of crashing immediately; fail only if <2 valid agents remain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path('.')
DATA_PATH = ROOT / 'data_processed' / 'cicids2017_all.parquet'
MODEL_PATHS = {
    'random': ROOT / 'models' /'step2'/ 'step2_agents_random.joblib',
    'aap': ROOT / 'models' /'step2'/ 'step2_agents_aap.joblib',

    'supervised': ROOT / 'models' / 'local_agents_supervised.joblib',
}
OUTDIR = ROOT / 'results_cross_partition_fair'
OUTDIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TARGET_FPR = 0.01
TRIM_K = 1


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.sum(y_true) == 0:
        return float('nan')
    return float(average_precision_score(y_true, y_score))


def threshold_from_negatives(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = TARGET_FPR) -> float:
    neg = y_score[y_true == 0]
    if len(neg) == 0:
        return 1.0
    q = min(max(1.0 - target_fpr, 0.0), 1.0)
    return float(np.quantile(neg, q))


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


def eval_with_calibration(y_va: np.ndarray, p_va: np.ndarray, y_te: np.ndarray, p_te: np.ndarray,
                          y_cb: np.ndarray, p_cb: np.ndarray, target_fpr: float = TARGET_FPR) -> Dict[str, float]:
    thr = threshold_from_negatives(y_cb, p_cb, target_fpr=target_fpr)
    return {
        'roc_auc': safe_roc_auc(y_te, p_te),
        'pr_auc': safe_pr_auc(y_te, p_te),
        'tpr_at_fpr1': tpr_at_threshold(y_te, p_te, thr),
        'monitor_fpr': fpr_at_threshold(y_cb, p_cb, thr),
        'decision_threshold': float(thr),
        'val_pr_auc': safe_pr_auc(y_va, p_va),
        'val_roc_auc': safe_roc_auc(y_va, p_va),
    }


def robust_attention_weights(score_va: np.ndarray, score_cb: np.ndarray, y_va: np.ndarray) -> np.ndarray:
    aps = np.array([safe_pr_auc(y_va, score_va[:, i]) for i in range(score_va.shape[1])], dtype=float)
    aps = np.nan_to_num(aps, nan=0.0)
    base = np.exp(aps / 0.1)
    base = base / np.clip(base.sum(), 1e-12, None)

    mu_ref = score_va.mean(axis=0)
    sd_ref = score_va.std(axis=0) + 1e-6
    mu_cur = score_cb.mean(axis=0)
    sd_cur = score_cb.std(axis=0) + 1e-6
    shift = np.abs(mu_cur - mu_ref) / sd_ref + 0.5 * np.abs(sd_cur - sd_ref) / sd_ref
    health = np.exp(-shift)
    w = base * health
    w = w / np.clip(w.sum(), 1e-12, None)
    return w


def pair_weight_from_validation(p_va_i: np.ndarray, p_va_j: np.ndarray, y_va: np.ndarray) -> np.ndarray:
    ap_i = safe_pr_auc(y_va, p_va_i)
    ap_j = safe_pr_auc(y_va, p_va_j)
    w = np.exp(np.array([ap_i, ap_j]) / 0.1)
    w = w / np.clip(w.sum(), 1e-12, None)
    return w


def pair_fuse_mean(p_i: np.ndarray, p_j: np.ndarray) -> np.ndarray:
    return 0.5 * (p_i + p_j)


def pair_fuse_weighted(p_i: np.ndarray, p_j: np.ndarray, w: np.ndarray) -> np.ndarray:
    return w[0] * p_i + w[1] * p_j


def trim_mean(scores: np.ndarray, trim_k: int = TRIM_K) -> np.ndarray:
    if scores.shape[1] <= 2 * trim_k:
        return scores.mean(axis=1)
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, trim_k:-trim_k].mean(axis=1)


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_days = {'tuesday', 'wednesday', 'thursday'}
    df = df.copy()
    df['day'] = df['day'].astype(str).str.strip().str.lower()
    train_df = df[df['day'].isin(train_days)].copy()
    test_df = df[df['day'] == 'friday'].copy()
    monday_df = df[df['day'] == 'monday'].copy()
    from sklearn.model_selection import train_test_split
    if len(train_df) == 0 or len(test_df) == 0 or len(monday_df) == 0:
        raise ValueError(f"Day split empty after normalization: train={train_df.shape}, test={test_df.shape}, monday={monday_df.shape}. Unique day values: {sorted(df['day'].dropna().unique().tolist())[:20]}")
    _, va_df = train_test_split(train_df, test_size=0.2, stratify=train_df['y'], random_state=SEED)
    ca_df, cb_df = train_test_split(monday_df, test_size=0.5, stratify=monday_df['y'], random_state=SEED)
    return train_df, va_df.reset_index(drop=True), ca_df.reset_index(drop=True), cb_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_agent_bundle(path: Path):
    obj = joblib.load(path)
    agents = obj['agents'] if isinstance(obj, dict) and 'agents' in obj else obj
    if not isinstance(agents, dict):
        raise ValueError(f'Unexpected model format: {path}')
    return agents


def normalize_cols(cols) -> List[str]:
    if isinstance(cols, str):
        out = [cols]
    elif isinstance(cols, pd.Index):
        out = cols.astype(str).tolist()
    elif isinstance(cols, np.ndarray):
        if cols.ndim == 0:
            out = [str(cols.item())]
        else:
            out = [str(x) for x in cols.tolist()]
    elif isinstance(cols, (list, tuple, set)):
        out = [str(x) for x in list(cols)]
    else:
        out = [str(cols)]
    out = [c for c in out if isinstance(c, str) and c != '']
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq

def build_numeric_matrix(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    """Return a strictly 2D numeric matrix for LightGBM prediction."""
    X = df.loc[:, cols].copy()
    # Coerce every column to numeric explicitly; invalid parsing becomes NaN.
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = X.to_numpy(dtype=np.float32, copy=False)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr



def predict_agent_scores(mode: str, agents: Dict[str, dict], df_va: pd.DataFrame, df_te: pd.DataFrame, df_cb: pd.DataFrame):
    names = sorted(agents.keys())
    score_va = []
    score_te = []
    score_cb = []
    rows = []
    skipped = []
    for name in names:
        bundle = agents[name]
        model = bundle['model']
        cols = normalize_cols(bundle.get('cols', []))
        if not cols:
            skipped.append({'agent': name, 'reason': 'empty_cols'})
            continue
        present = [c for c in cols if c in df_va.columns and c in df_te.columns and c in df_cb.columns]
        missing = [c for c in cols if c not in present]
        if not present:
            skipped.append({'agent': name, 'reason': 'all_cols_missing', 'missing_cols': missing})
            continue
        try:
            x_va = build_numeric_matrix(df_va, present)
            x_te = build_numeric_matrix(df_te, present)
            x_cb = build_numeric_matrix(df_cb, present)
            if x_va.ndim != 2 or x_va.shape[1] == 0:
                skipped.append({'agent': name, 'reason': 'empty_2d_after_select', 'present_cols': present})
                continue
            p_va = model.predict_proba(x_va)[:, 1]
            p_te = model.predict_proba(x_te)[:, 1]
            p_cb = model.predict_proba(x_cb)[:, 1]
        except Exception as exc:
            skipped.append({
                'agent': name,
                'reason': f'predict_failed: {exc}',
                'present_cols': present,
                'missing_cols': missing,
                'n_present_cols': len(present),
            })
            continue
        score_va.append(p_va)
        score_te.append(p_te)
        score_cb.append(p_cb)
        rows.append({'agent': name, 'cols': present, 'missing_cols': missing, 'p_va': p_va, 'p_te': p_te, 'p_cb': p_cb})
    if skipped:
        pd.DataFrame(skipped).to_csv(OUTDIR / f'fair_compare_invalid_agents_{mode}.csv', index=False)
    if len(rows) < 2:
        raise ValueError(f'{mode}: fewer than 2 valid agents remain after normalization. Details written to fair_compare_invalid_agents_{mode}.csv')
    valid_names = [r['agent'] for r in rows]
    return valid_names, np.column_stack(score_va), np.column_stack(score_te), np.column_stack(score_cb), rows


def pick_best_single(rows: List[dict], y_va: np.ndarray) -> str:
    ranked = []
    for r in rows:
        ranked.append((r['agent'], safe_pr_auc(y_va, r['p_va']), safe_roc_auc(y_va, r['p_va'])))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return ranked[0][0]


def build_full_methods(score_va: np.ndarray, score_te: np.ndarray, score_cb: np.ndarray, y_va: np.ndarray):
    methods = {}
    methods['Global_Full_L2_mean'] = {'p_va': score_va.mean(axis=1), 'p_te': score_te.mean(axis=1), 'p_cb': score_cb.mean(axis=1), 'comm_cost': score_va.shape[1], 'note': 'all agents mean'}
    methods['Global_Full_L2_trimmed'] = {'p_va': trim_mean(score_va), 'p_te': trim_mean(score_te), 'p_cb': trim_mean(score_cb), 'comm_cost': score_va.shape[1], 'note': 'all agents trimmed mean'}
    w = robust_attention_weights(score_va, score_cb, y_va)
    methods['Global_Full_L2_attention_robust'] = {'p_va': score_va @ w, 'p_te': score_te @ w, 'p_cb': score_cb @ w, 'comm_cost': score_va.shape[1], 'note': 'all agents robust attention'}
    try:
        meta = LogisticRegression(max_iter=1000, random_state=SEED)
        meta.fit(score_va, y_va)
        methods['Global_Full_L2_stacking_lr'] = {'p_va': meta.predict_proba(score_va)[:, 1], 'p_te': meta.predict_proba(score_te)[:, 1], 'p_cb': meta.predict_proba(score_cb)[:, 1], 'comm_cost': score_va.shape[1], 'note': 'all agents stacking lr'}
    except Exception as exc:
        methods['Global_Full_L2_stacking_lr'] = {'p_va': score_va.mean(axis=1), 'p_te': score_te.mean(axis=1), 'p_cb': score_cb.mean(axis=1), 'comm_cost': score_va.shape[1], 'note': f'stacking failed: {exc}'}
    return methods


def pair_search(rows: List[dict], y_va: np.ndarray):
    pair_rank_rows = []
    best_weighted = None
    best_mean = None
    best_weighted_key = None
    best_mean_key = None
    agent_names = [r['agent'] for r in rows]
    by_name = {r['agent']: r for r in rows}
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            a, b = agent_names[i], agent_names[j]
            ra, rb = by_name[a], by_name[b]
            w = pair_weight_from_validation(ra['p_va'], rb['p_va'], y_va)
            pw_va = pair_fuse_weighted(ra['p_va'], rb['p_va'], w)
            pw_te = pair_fuse_weighted(ra['p_te'], rb['p_te'], w)
            pw_cb = pair_fuse_weighted(ra['p_cb'], rb['p_cb'], w)
            pm_va = pair_fuse_mean(ra['p_va'], rb['p_va'])
            pm_te = pair_fuse_mean(ra['p_te'], rb['p_te'])
            pm_cb = pair_fuse_mean(ra['p_cb'], rb['p_cb'])
            val_pr_w = safe_pr_auc(y_va, pw_va)
            val_pr_m = safe_pr_auc(y_va, pm_va)
            val_roc_w = safe_roc_auc(y_va, pw_va)
            val_roc_m = safe_roc_auc(y_va, pm_va)
            pair_rank_rows.append({'pair': f'{a}+{b}', 'agent_a': a, 'agent_b': b, 'val_pr_auc_weighted': val_pr_w, 'val_pr_auc_mean': val_pr_m, 'val_roc_auc_weighted': val_roc_w, 'val_roc_auc_mean': val_roc_m, 'w_a': float(w[0]), 'w_b': float(w[1])})
            key_w = (val_pr_w, val_roc_w)
            key_m = (val_pr_m, val_roc_m)
            if best_weighted is None or key_w > best_weighted_key:
                best_weighted_key = key_w
                best_weighted = {'pair': f'{a}+{b}', 'p_va': pw_va, 'p_te': pw_te, 'p_cb': pw_cb, 'comm_cost': 2, 'note': f'weights={w.tolist()}'}
            if best_mean is None or key_m > best_mean_key:
                best_mean_key = key_m
                best_mean = {'pair': f'{a}+{b}', 'p_va': pm_va, 'p_te': pm_te, 'p_cb': pm_cb, 'comm_cost': 2, 'note': 'simple mean'}
    pair_rank_df = pd.DataFrame(pair_rank_rows).sort_values('val_pr_auc_weighted', ascending=False)
    return best_weighted, best_mean, pair_rank_df


def run_partition(mode: str):
    df = pd.read_parquet(DATA_PATH)
    _, va_df, ca_df, cb_df, te_df = split_data(df)
    y_va = va_df['y'].to_numpy().astype(int)
    y_cb = cb_df['y'].to_numpy().astype(int)
    y_te = te_df['y'].to_numpy().astype(int)
    agents = load_agent_bundle(MODEL_PATHS[mode])
    _, score_va, score_te, score_cb, rows = predict_agent_scores(mode, agents, va_df, te_df, cb_df)

    records = []
    best_single_name = pick_best_single(rows, y_va)
    by_name = {r['agent']: r for r in rows}
    r = by_name[best_single_name]
    metrics = eval_with_calibration(y_va, r['p_va'], y_te, r['p_te'], y_cb, r['p_cb'])
    records.append({'partition_mode': mode, 'method': 'Global_Best_L0', 'comm_cost': 1, 'note': best_single_name, **metrics})

    best_weighted, best_mean, pair_rank_df = pair_search(rows, y_va)
    for meth_name, obj in [('Global_Best_Pair_L2_weighted', best_weighted), ('Global_Best_Pair_L2_mean', best_mean)]:
        metrics = eval_with_calibration(y_va, obj['p_va'], y_te, obj['p_te'], y_cb, obj['p_cb'])
        records.append({'partition_mode': mode, 'method': meth_name, 'comm_cost': obj['comm_cost'], 'note': obj['pair'] + ' | ' + obj['note'], **metrics})

    full_methods = build_full_methods(score_va, score_te, score_cb, y_va)
    for meth_name, obj in full_methods.items():
        metrics = eval_with_calibration(y_va, obj['p_va'], y_te, obj['p_te'], y_cb, obj['p_cb'])
        records.append({'partition_mode': mode, 'method': meth_name, 'comm_cost': obj['comm_cost'], 'note': obj['note'], **metrics})

    result_df = pd.DataFrame(records)
    pair_rank_df.to_csv(OUTDIR / f'fair_pair_rank_weighted_{mode}.csv', index=False)
    result_df.to_csv(OUTDIR / f'fair_compare_{mode}.csv', index=False)
    return result_df


def make_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for mode, sub in all_df.groupby('partition_mode'):
        get = lambda m, c: float(sub.loc[sub['method'] == m, c].iloc[0]) if (sub['method'] == m).any() else float('nan')
        get_note = lambda m: str(sub.loc[sub['method'] == m, 'note'].iloc[0]) if (sub['method'] == m).any() else ''
        row = {
            'partition_mode': mode,
            'best_l0_agent': get_note('Global_Best_L0'),
            'best_pair_weighted': get_note('Global_Best_Pair_L2_weighted'),
            'best_pair_mean': get_note('Global_Best_Pair_L2_mean'),
            'l0_tpr': get('Global_Best_L0', 'tpr_at_fpr1'),
            'pair_weighted_tpr': get('Global_Best_Pair_L2_weighted', 'tpr_at_fpr1'),
            'pair_mean_tpr': get('Global_Best_Pair_L2_mean', 'tpr_at_fpr1'),
            'full_mean_tpr': get('Global_Full_L2_mean', 'tpr_at_fpr1'),
            'full_trimmed_tpr': get('Global_Full_L2_trimmed', 'tpr_at_fpr1'),
            'full_robust_tpr': get('Global_Full_L2_attention_robust', 'tpr_at_fpr1'),
            'l0_pr': get('Global_Best_L0', 'pr_auc'),
            'pair_weighted_pr': get('Global_Best_Pair_L2_weighted', 'pr_auc'),
            'pair_mean_pr': get('Global_Best_Pair_L2_mean', 'pr_auc'),
            'gain_pair_weighted_vs_l0': get('Global_Best_Pair_L2_weighted', 'tpr_at_fpr1') - get('Global_Best_L0', 'tpr_at_fpr1'),
            'gain_pair_mean_vs_l0': get('Global_Best_Pair_L2_mean', 'tpr_at_fpr1') - get('Global_Best_L0', 'tpr_at_fpr1'),
            'gain_full_robust_vs_l0': get('Global_Full_L2_attention_robust', 'tpr_at_fpr1') - get('Global_Best_L0', 'tpr_at_fpr1'),
        }
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values('partition_mode').reset_index(drop=True)
    summary.to_csv(OUTDIR / 'fair_compare_summary.csv', index=False)
    return summary


def make_plots(all_df: pd.DataFrame):
    focus_methods = ['Global_Best_L0', 'Global_Best_Pair_L2_weighted', 'Global_Full_L2_mean', 'Global_Full_L2_attention_robust']
    focus = all_df[all_df['method'].isin(focus_methods)].copy()
    focus['label'] = focus['partition_mode'] + '\n' + focus['method'].str.replace('Global_', '', regex=False)
    plt.figure(figsize=(12, 5))
    xs = np.arange(len(focus))
    plt.bar(xs, focus['tpr_at_fpr1'].values)
    plt.xticks(xs, focus['label'].values, rotation=45, ha='right')
    plt.ylabel('TPR@1%FPR')
    plt.title('Cross-partition fair comparison: TPR@1%FPR')
    plt.tight_layout()
    plt.savefig(OUTDIR / 'fair_compare_tpr_bar.png', dpi=200)
    plt.close()
    plt.figure(figsize=(12, 5))
    xs = np.arange(len(focus))
    plt.bar(xs, focus['comm_cost'].values)
    plt.xticks(xs, focus['label'].values, rotation=45, ha='right')
    plt.ylabel('Communication cost')
    plt.title('Cross-partition fair comparison: communication cost')
    plt.tight_layout()
    plt.savefig(OUTDIR / 'fair_compare_cost_bar.png', dpi=200)
    plt.close()
    plt.figure(figsize=(8, 6))
    for mode, sub in focus.groupby('partition_mode'):
        plt.scatter(sub['comm_cost'], sub['tpr_at_fpr1'], label=mode, s=60)
        for _, r in sub.iterrows():
            short = r['method'].replace('Global_', '').replace('L2_', '')
            plt.annotate(f'{mode}:{short}', (r['comm_cost'], r['tpr_at_fpr1']), fontsize=8)
    plt.xlabel('Communication cost')
    plt.ylabel('TPR@1%FPR')
    plt.title('Cross-partition fair comparison: TPR vs cost')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / 'fair_compare_tpr_vs_cost.png', dpi=200)
    plt.close()


def write_markdown(summary: pd.DataFrame):
    lines = ['# Cross-partition fair comparison summary', '']
    for _, r in summary.iterrows():
        lines.append(f"## {r['partition_mode']}")
        lines.append(f"- Best L0: {r['best_l0_agent']} | TPR@1%FPR={r['l0_tpr']:.6f} | PR-AUC={r['l0_pr']:.6f}")
        lines.append(f"- Best Pair (weighted): {r['best_pair_weighted']} | TPR@1%FPR={r['pair_weighted_tpr']:.6f} | PR-AUC={r['pair_weighted_pr']:.6f}")
        lines.append(f"- Best Pair gain vs L0: {r['gain_pair_weighted_vs_l0']:+.6f}")
        lines.append(f"- Full robust gain vs L0: {r['gain_full_robust_vs_l0']:+.6f}")
        lines.append('')
    (OUTDIR / 'fair_compare_summary.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    missing = [str(p) for p in [DATA_PATH, *MODEL_PATHS.values()] if not p.exists()]
    if missing:
        raise FileNotFoundError('Missing required files:\n' + '\n'.join(missing))
    all_parts = []
    for mode in ['random', 'aap', 'supervised']:
        df_mode = run_partition(mode)
        all_parts.append(df_mode)
    all_df = pd.concat(all_parts, ignore_index=True)
    all_df.to_csv(OUTDIR / 'fair_compare_all.csv', index=False)
    summary = make_summary(all_df)
    write_markdown(summary)
    make_plots(all_df)
    print('[OK] Wrote results to', OUTDIR)


if __name__ == '__main__':
    main()
