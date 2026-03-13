from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = 'data_processed/cicids2017_all.parquet'
MODEL_DIR = 'models'
OUTPUT_DIR = 'results_fair_compare'
RANDOM_SEED = 42
TARGET_FPR = 0.01
EPS = 1e-6
PARTITIONS = ['random', 'aap', 'supervised']


@dataclass
class EvalResult:
    method: str
    partition_mode: str
    avg_comm_cost: float
    test_roc_auc: float
    test_pr_auc: float
    test_tpr_at_fpr1: float
    monitor_fpr: float
    decision_threshold: float
    val_proxy_tpr: float
    selection_note: str


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
        return float('nan')
    return float((p_score[neg] >= threshold).mean())


def rowwise_trimmed_mean(mat: np.ndarray, trim_k: int = 1) -> np.ndarray:
    if mat.shape[1] <= 2 * trim_k:
        return mat.mean(axis=1)
    sm = np.sort(mat, axis=1)
    return sm[:, trim_k:-trim_k].mean(axis=1)


def compute_agent_health(ref_mat: np.ndarray, cur_mat: np.ndarray) -> np.ndarray:
    ref_mean = ref_mat.mean(axis=0)
    cur_mean = cur_mat.mean(axis=0)
    ref_std = ref_mat.std(axis=0) + EPS
    cur_std = cur_mat.std(axis=0)
    shift = np.abs(cur_mean - ref_mean) / ref_std + 0.5 * np.abs(cur_std - ref_std) / ref_std
    health = np.exp(-shift)
    return np.clip(health, 0.05, 1.0)


def evaluate_scores(method: str, partition_mode: str, p_test: np.ndarray, p_calib_threshold: np.ndarray,
                    p_calib_monitor: np.ndarray, y_test: np.ndarray, y_calib_monitor: np.ndarray,
                    avg_comm_cost: float, val_proxy_tpr: float, selection_note: str) -> EvalResult:
    thr = threshold_from_calib(p_calib_threshold)
    return EvalResult(
        method=method,
        partition_mode=partition_mode,
        avg_comm_cost=float(avg_comm_cost),
        test_roc_auc=float(roc_auc_score(y_test, p_test)),
        test_pr_auc=float(average_precision_score(y_test, p_test)),
        test_tpr_at_fpr1=tpr_at_threshold(y_test, p_test, thr),
        monitor_fpr=fpr_at_threshold(y_calib_monitor, p_calib_monitor, thr),
        decision_threshold=thr,
        val_proxy_tpr=float(val_proxy_tpr),
        selection_note=selection_note,
    )


def load_day_splits(df: pd.DataFrame):
    train_days = {'tuesday', 'wednesday', 'thursday'}
    train_df = df[df['day'].isin(train_days)].copy()
    test_df = df[df['day'] == 'friday'].copy()
    monday_df = df[df['day'] == 'monday'].copy()
    _, va_df = train_test_split(train_df, test_size=0.2, stratify=train_df['y'], random_state=RANDOM_SEED)
    cal_a_df, cal_b_df = train_test_split(monday_df, test_size=0.5, stratify=monday_df['y'], random_state=RANDOM_SEED)
    return va_df, test_df, cal_a_df, cal_b_df


def build_agent_rows(agents: Dict, va_df: pd.DataFrame, test_df: pd.DataFrame, cal_a_df: pd.DataFrame, cal_b_df: pd.DataFrame):
    rows = []
    for name, bundle in agents.items():
        model = bundle['model']
        cols = bundle['cols']
        rows.append({
            'agent': name,
            'cols': cols,
            'p_va': model.predict_proba(va_df[cols])[:, 1],
            'p_te': model.predict_proba(test_df[cols])[:, 1],
            'p_ca': model.predict_proba(cal_a_df[cols])[:, 1],
            'p_cb': model.predict_proba(cal_b_df[cols])[:, 1],
        })
    return rows


def select_best_single(rows: List[Dict], y_va: np.ndarray):
    rank_rows = []
    best = None
    best_key = None
    for row in rows:
        thr = threshold_from_calib(row['p_ca'])
        val_tpr = tpr_at_threshold(y_va, row['p_va'], thr)
        val_pr = float(average_precision_score(y_va, row['p_va']))
        val_roc = float(roc_auc_score(y_va, row['p_va']))
        rank_rows.append({
            'agent': row['agent'],
            'val_proxy_tpr': val_tpr,
            'val_pr_auc': val_pr,
            'val_roc_auc': val_roc,
            'val_threshold': thr,
            'feature_count': len(row['cols']),
        })
        key = (val_tpr, val_pr, val_roc)
        if best_key is None or key > best_key:
            best_key = key
            best = row
    rank_df = pd.DataFrame(rank_rows).sort_values(['val_proxy_tpr', 'val_pr_auc', 'val_roc_auc'], ascending=False)
    return best, rank_df


def fuse_pair(a: Dict, b: Dict, y_va: np.ndarray, weighted: bool = True, temp: float = 0.1):
    if weighted:
        aps = np.array([
            average_precision_score(y_va, a['p_va']),
            average_precision_score(y_va, b['p_va']),
        ])
        w = softmax(aps / temp)
        note = f"{a['agent']}+{b['agent']}|w={w[0]:.3f},{w[1]:.3f}"
    else:
        w = np.array([0.5, 0.5])
        note = f"{a['agent']}+{b['agent']}|mean"
    return {
        'agents': (a['agent'], b['agent']),
        'weights': w,
        'note': note,
        'p_va': w[0] * a['p_va'] + w[1] * b['p_va'],
        'p_te': w[0] * a['p_te'] + w[1] * b['p_te'],
        'p_ca': w[0] * a['p_ca'] + w[1] * b['p_ca'],
        'p_cb': w[0] * a['p_cb'] + w[1] * b['p_cb'],
    }


def select_best_pair(rows: List[Dict], y_va: np.ndarray, weighted: bool = True):
    pair_rows = []
    best = None
    best_key = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            fused = fuse_pair(rows[i], rows[j], y_va, weighted=weighted)
            thr = threshold_from_calib(fused['p_ca'])
            val_tpr = tpr_at_threshold(y_va, fused['p_va'], thr)
            val_pr = float(average_precision_score(y_va, fused['p_va']))
            val_roc = float(roc_auc_score(y_va, fused['p_va']))
            pair_rows.append({
                'agent_a': fused['agents'][0],
                'agent_b': fused['agents'][1],
                'weighted': weighted,
                'val_proxy_tpr': val_tpr,
                'val_pr_auc': val_pr,
                'val_roc_auc': val_roc,
                'val_threshold': thr,
                'selection_note': fused['note'],
            })
            key = (val_tpr, val_pr, val_roc)
            if best_key is None or key > best_key:
                best_key = key
                best = fused
    rank_df = pd.DataFrame(pair_rows).sort_values(['val_proxy_tpr', 'val_pr_auc', 'val_roc_auc'], ascending=False)
    return best, rank_df


def run_partition(mode: str, data_path: str = DATA_PATH, model_dir: str = MODEL_DIR):
    model_path = Path(model_dir) / f'local_agents_{mode}.joblib'
    if not model_path.exists():
        return [], None, None, None

    df = pd.read_parquet(data_path)
    va_df, test_df, cal_a_df, cal_b_df = load_day_splits(df)
    y_va = va_df['y'].values
    y_te = test_df['y'].values
    y_cb = cal_b_df['y'].values

    agents = joblib.load(model_path)
    rows = build_agent_rows(agents, va_df, test_df, cal_a_df, cal_b_df)
    mats = {
        'va': np.column_stack([r['p_va'] for r in rows]),
        'te': np.column_stack([r['p_te'] for r in rows]),
        'ca': np.column_stack([r['p_ca'] for r in rows]),
        'cb': np.column_stack([r['p_cb'] for r in rows]),
    }

    results = []

    best_single, single_rank = select_best_single(rows, y_va)
    best_single_thr = threshold_from_calib(best_single['p_ca'])
    best_single_val_tpr = tpr_at_threshold(y_va, best_single['p_va'], best_single_thr)
    results.append(evaluate_scores(
        'Global_Best_L0', mode.upper(), best_single['p_te'], best_single['p_ca'], best_single['p_cb'], y_te, y_cb,
        1, best_single_val_tpr, best_single['agent']
    ))

    best_pair_weighted, pair_rank_weighted = select_best_pair(rows, y_va, weighted=True)
    thr_w = threshold_from_calib(best_pair_weighted['p_ca'])
    val_tpr_w = tpr_at_threshold(y_va, best_pair_weighted['p_va'], thr_w)
    results.append(evaluate_scores(
        'Global_Best_Pair_L2_weighted', mode.upper(), best_pair_weighted['p_te'], best_pair_weighted['p_ca'], best_pair_weighted['p_cb'],
        y_te, y_cb, 2, val_tpr_w, best_pair_weighted['note']
    ))

    best_pair_mean, pair_rank_mean = select_best_pair(rows, y_va, weighted=False)
    thr_m = threshold_from_calib(best_pair_mean['p_ca'])
    val_tpr_m = tpr_at_threshold(y_va, best_pair_mean['p_va'], thr_m)
    results.append(evaluate_scores(
        'Global_Best_Pair_L2_mean', mode.upper(), best_pair_mean['p_te'], best_pair_mean['p_ca'], best_pair_mean['p_cb'],
        y_te, y_cb, 2, val_tpr_m, best_pair_mean['note']
    ))

    p_va_mean = mats['va'].mean(axis=1)
    p_te_mean = mats['te'].mean(axis=1)
    p_ca_mean = mats['ca'].mean(axis=1)
    p_cb_mean = mats['cb'].mean(axis=1)
    thr_mean = threshold_from_calib(p_ca_mean)
    results.append(evaluate_scores(
        'Global_Full_L2_mean', mode.upper(), p_te_mean, p_ca_mean, p_cb_mean, y_te, y_cb, mats['te'].shape[1],
        tpr_at_threshold(y_va, p_va_mean, thr_mean), 'all_agents_mean'
    ))

    p_va_trim = rowwise_trimmed_mean(mats['va'], 1)
    p_te_trim = rowwise_trimmed_mean(mats['te'], 1)
    p_ca_trim = rowwise_trimmed_mean(mats['ca'], 1)
    p_cb_trim = rowwise_trimmed_mean(mats['cb'], 1)
    thr_trim = threshold_from_calib(p_ca_trim)
    results.append(evaluate_scores(
        'Global_Full_L2_trimmed', mode.upper(), p_te_trim, p_ca_trim, p_cb_trim, y_te, y_cb, mats['te'].shape[1],
        tpr_at_threshold(y_va, p_va_trim, thr_trim), 'all_agents_trimmed_mean'
    ))

    ap_va = np.array([average_precision_score(y_va, mats['va'][:, i]) for i in range(mats['va'].shape[1])])
    base_w = softmax(ap_va / 0.1)
    health = compute_agent_health(mats['va'], mats['ca'])
    robust_w = base_w * health
    robust_w = robust_w / robust_w.sum()
    p_va_rob = mats['va'] @ robust_w
    p_te_rob = mats['te'] @ robust_w
    p_ca_rob = mats['ca'] @ robust_w
    p_cb_rob = mats['cb'] @ robust_w
    thr_rob = threshold_from_calib(p_ca_rob)
    results.append(evaluate_scores(
        'Global_Full_L2_attention_robust', mode.upper(), p_te_rob, p_ca_rob, p_cb_rob, y_te, y_cb, mats['te'].shape[1],
        tpr_at_threshold(y_va, p_va_rob, thr_rob), 'robust_attention'
    ))

    try:
        meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
        meta.fit(mats['va'], y_va)
        p_va_stack = meta.predict_proba(mats['va'])[:, 1]
        p_te_stack = meta.predict_proba(mats['te'])[:, 1]
        p_ca_stack = meta.predict_proba(mats['ca'])[:, 1]
        p_cb_stack = meta.predict_proba(mats['cb'])[:, 1]
        thr_stack = threshold_from_calib(p_ca_stack)
        results.append(evaluate_scores(
            'Global_Full_L2_stacking_lr', mode.upper(), p_te_stack, p_ca_stack, p_cb_stack, y_te, y_cb, mats['te'].shape[1],
            tpr_at_threshold(y_va, p_va_stack, thr_stack), 'stacking_lr'
        ))
    except Exception:
        pass

    return results, single_rank, pair_rank_weighted, pair_rank_mean


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    summary_rows = []

    for mode in PARTITIONS:
        results, single_rank, pair_rank_w, pair_rank_m = run_partition(mode)
        if not results:
            continue
        df_part = pd.DataFrame([asdict(r) for r in results])
        df_part.to_csv(out_dir / f'fair_compare_{mode}.csv', index=False)
        if single_rank is not None:
            single_rank.to_csv(out_dir / f'fair_single_rank_{mode}.csv', index=False)
        if pair_rank_w is not None:
            pair_rank_w.to_csv(out_dir / f'fair_pair_rank_weighted_{mode}.csv', index=False)
        if pair_rank_m is not None:
            pair_rank_m.to_csv(out_dir / f'fair_pair_rank_mean_{mode}.csv', index=False)
        all_results.extend([asdict(r) for r in results])

        part = df_part.set_index('method')
        if 'Global_Best_L0' in part.index and 'Global_Best_Pair_L2_weighted' in part.index:
            summary_rows.append({
                'partition_mode': mode.upper(),
                'best_l0_tpr': part.loc['Global_Best_L0', 'test_tpr_at_fpr1'],
                'best_pair_weighted_tpr': part.loc['Global_Best_Pair_L2_weighted', 'test_tpr_at_fpr1'],
                'gain_weighted_vs_l0': part.loc['Global_Best_Pair_L2_weighted', 'test_tpr_at_fpr1'] - part.loc['Global_Best_L0', 'test_tpr_at_fpr1'],
                'best_pair_mean_tpr': part.loc['Global_Best_Pair_L2_mean', 'test_tpr_at_fpr1'],
                'gain_mean_vs_l0': part.loc['Global_Best_Pair_L2_mean', 'test_tpr_at_fpr1'] - part.loc['Global_Best_L0', 'test_tpr_at_fpr1'],
                'best_l0_pr': part.loc['Global_Best_L0', 'test_pr_auc'],
                'best_pair_weighted_pr': part.loc['Global_Best_Pair_L2_weighted', 'test_pr_auc'],
                'best_pair_mean_pr': part.loc['Global_Best_Pair_L2_mean', 'test_pr_auc'],
                'best_l0_note': part.loc['Global_Best_L0', 'selection_note'],
                'best_pair_weighted_note': part.loc['Global_Best_Pair_L2_weighted', 'selection_note'],
                'best_pair_mean_note': part.loc['Global_Best_Pair_L2_mean', 'selection_note'],
            })

    all_df = pd.DataFrame(all_results)
    if not all_df.empty:
        all_df.to_csv(out_dir / 'fair_compare_all.csv', index=False)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.to_csv(out_dir / 'fair_compare_summary.csv', index=False)
        lines = ['# Fair Single vs Pair Comparison', '']
        for _, r in summary_df.iterrows():
            lines.append(f"## {r['partition_mode']}")
            lines.append(f"- Global_Best_L0 TPR@1%FPR: {r['best_l0_tpr']:.6f} ({r['best_l0_note']})")
            lines.append(f"- Global_Best_Pair_L2_weighted TPR@1%FPR: {r['best_pair_weighted_tpr']:.6f} ({r['best_pair_weighted_note']})")
            lines.append(f"- Gain vs L0: {r['gain_weighted_vs_l0']:+.6f}")
            lines.append(f"- Global_Best_Pair_L2_mean TPR@1%FPR: {r['best_pair_mean_tpr']:.6f} ({r['best_pair_mean_note']})")
            lines.append(f"- Gain vs L0: {r['gain_mean_vs_l0']:+.6f}")
            lines.append('')
        (out_dir / 'fair_compare_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    print(f'[OK] wrote results to {out_dir}')


if __name__ == '__main__':
    main()
