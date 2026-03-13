from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = 'data_processed/cicids2017_all.parquet'
MODEL_DIR = 'models'
OUTPUT_DIR = 'results_best_pair_compare'
RANDOM_SEED = 42
TARGET_FPR = 0.01
EPS = 1e-6
PARTITIONS = ['random', 'aap', 'supervised']
FOCUS_METHODS = [
    'Best_L0',
    'Best_Pair_L2_weighted',
    'Best_Pair_L2_mean',
    'L2_mean',
    'L2_trimmed_mean',
    'L2_attention_robust',
]


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


def select_best_l0(rows: List[Dict], y_va: np.ndarray):
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
        tag = f"{a['agent']}+{b['agent']}|w={w[0]:.3f},{w[1]:.3f}"
    else:
        w = np.array([0.5, 0.5])
        tag = f"{a['agent']}+{b['agent']}|mean"
    return {
        'agents': (a['agent'], b['agent']),
        'weights': w,
        'note': tag,
        'p_va': w[0] * a['p_va'] + w[1] * b['p_va'],
        'p_te': w[0] * a['p_te'] + w[1] * b['p_te'],
        'p_ca': w[0] * a['p_ca'] + w[1] * b['p_ca'],
        'p_cb': w[0] * a['p_cb'] + w[1] * b['p_cb'],
    }


def rank_pairs(rows: List[Dict], y_va: np.ndarray, weighted: bool = True):
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
    pair_df = pd.DataFrame(pair_rows).sort_values(['val_proxy_tpr', 'val_pr_auc', 'val_roc_auc'], ascending=False)
    return best, pair_df


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

    best_l0, l0_rank = select_best_l0(rows, y_va)
    best_l0_thr = threshold_from_calib(best_l0['p_ca'])
    best_l0_val_tpr = tpr_at_threshold(y_va, best_l0['p_va'], best_l0_thr)
    results.append(evaluate_scores(
        'Best_L0', mode.upper(), best_l0['p_te'], best_l0['p_ca'], best_l0['p_cb'], y_te, y_cb,
        1, best_l0_val_tpr, best_l0['agent']
    ))

    best_pair_weighted, pair_rank_weighted = rank_pairs(rows, y_va, weighted=True)
    best_pair_w_thr = threshold_from_calib(best_pair_weighted['p_ca'])
    best_pair_w_val_tpr = tpr_at_threshold(y_va, best_pair_weighted['p_va'], best_pair_w_thr)
    results.append(evaluate_scores(
        'Best_Pair_L2_weighted', mode.upper(), best_pair_weighted['p_te'], best_pair_weighted['p_ca'], best_pair_weighted['p_cb'],
        y_te, y_cb, 2, best_pair_w_val_tpr, best_pair_weighted['note']
    ))

    best_pair_mean, pair_rank_mean = rank_pairs(rows, y_va, weighted=False)
    best_pair_m_thr = threshold_from_calib(best_pair_mean['p_ca'])
    best_pair_m_val_tpr = tpr_at_threshold(y_va, best_pair_mean['p_va'], best_pair_m_thr)
    results.append(evaluate_scores(
        'Best_Pair_L2_mean', mode.upper(), best_pair_mean['p_te'], best_pair_mean['p_ca'], best_pair_mean['p_cb'],
        y_te, y_cb, 2, best_pair_m_val_tpr, best_pair_mean['note']
    ))

    p_te_mean = mats['te'].mean(axis=1)
    p_ca_mean = mats['ca'].mean(axis=1)
    p_cb_mean = mats['cb'].mean(axis=1)
    p_va_mean = mats['va'].mean(axis=1)
    results.append(evaluate_scores(
        'L2_mean', mode.upper(), p_te_mean, p_ca_mean, p_cb_mean, y_te, y_cb, mats['te'].shape[1],
        tpr_at_threshold(y_va, p_va_mean, threshold_from_calib(p_ca_mean)), 'all_agents_mean'
    ))

    p_te_trim = rowwise_trimmed_mean(mats['te'], 1)
    p_ca_trim = rowwise_trimmed_mean(mats['ca'], 1)
    p_cb_trim = rowwise_trimmed_mean(mats['cb'], 1)
    p_va_trim = rowwise_trimmed_mean(mats['va'], 1)
    results.append(evaluate_scores(
        'L2_trimmed_mean', mode.upper(), p_te_trim, p_ca_trim, p_cb_trim, y_te, y_cb, mats['te'].shape[1],
        tpr_at_threshold(y_va, p_va_trim, threshold_from_calib(p_ca_trim)), 'all_agents_trim1'
    ))

    va_aps = np.array([average_precision_score(y_va, r['p_va']) for r in rows])
    base_w = softmax(va_aps / 0.1)
    health = compute_agent_health(mats['va'], mats['ca'])
    robust_w = base_w * health
    robust_w = robust_w / robust_w.sum()
    p_va_rob = mats['va'] @ robust_w
    p_te_rob = mats['te'] @ robust_w
    p_ca_rob = mats['ca'] @ robust_w
    p_cb_rob = mats['cb'] @ robust_w
    selection_note = 'weights=' + ','.join(f'{rows[i]["agent"]}:{robust_w[i]:.3f}' for i in range(len(rows)))
    results.append(evaluate_scores(
        'L2_attention_robust', mode.upper(), p_te_rob, p_ca_rob, p_cb_rob, y_te, y_cb, mats['te'].shape[1],
        tpr_at_threshold(y_va, p_va_rob, threshold_from_calib(p_ca_rob)), selection_note
    ))

    # optional stacking baseline on all agents
    try:
        meta = LogisticRegression(max_iter=2000)
        meta.fit(mats['va'], y_va)
        p_va_stack = meta.predict_proba(mats['va'])[:, 1]
        p_te_stack = meta.predict_proba(mats['te'])[:, 1]
        p_ca_stack = meta.predict_proba(mats['ca'])[:, 1]
        p_cb_stack = meta.predict_proba(mats['cb'])[:, 1]
        results.append(evaluate_scores(
            'L2_stacking_lr', mode.upper(), p_te_stack, p_ca_stack, p_cb_stack, y_te, y_cb, mats['te'].shape[1],
            tpr_at_threshold(y_va, p_va_stack, threshold_from_calib(p_ca_stack)), 'all_agents_stacking'
        ))
    except Exception:
        pass

    return results, l0_rank, pair_rank_weighted, pair_rank_mean


def save_plots(summary_df: pd.DataFrame, out_dir: Path):
    focus = summary_df[summary_df['method'].isin(['Best_L0', 'Best_Pair_L2_weighted', 'Best_Pair_L2_mean'])].copy()
    if focus.empty:
        return

    plt.figure(figsize=(8, 4.6))
    x = np.arange(len(focus))
    labels = focus['partition_mode'] + '\n' + focus['method']
    plt.bar(x, focus['test_tpr_at_fpr1'])
    plt.xticks(x, labels, rotation=25, ha='right')
    plt.ylabel('TPR @ 1% FPR')
    plt.title('Best single-agent vs best pair')
    plt.tight_layout()
    plt.savefig(out_dir / 'best_pair_focus_tpr_bar.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.6))
    plt.bar(x, focus['avg_comm_cost'])
    plt.xticks(x, labels, rotation=25, ha='right')
    plt.ylabel('Communication cost')
    plt.title('Communication cost of best single-agent vs best pair')
    plt.tight_layout()
    plt.savefig(out_dir / 'best_pair_focus_cost_bar.png', dpi=200)
    plt.close()

    plt.figure(figsize=(6.4, 4.8))
    for method in ['Best_L0', 'Best_Pair_L2_weighted', 'Best_Pair_L2_mean']:
        sub = focus[focus['method'] == method]
        if len(sub):
            plt.scatter(sub['avg_comm_cost'], sub['test_tpr_at_fpr1'], s=70, label=method)
            for _, r in sub.iterrows():
                plt.annotate(r['partition_mode'], (r['avg_comm_cost'], r['test_tpr_at_fpr1']), textcoords='offset points', xytext=(4, 4))
    plt.xlabel('Communication cost')
    plt.ylabel('TPR @ 1% FPR')
    plt.title('Operational tradeoff: best single-agent vs best pair')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'best_pair_focus_tpr_vs_cost.png', dpi=200)
    plt.close()


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_l0_rank = []
    all_pair_rank_w = []
    all_pair_rank_m = []

    for mode in PARTITIONS:
        res, l0_rank, pair_rank_w, pair_rank_m = run_partition(mode)
        if res is None:
            continue
        all_results.extend([asdict(r) for r in res])
        if l0_rank is not None:
            l0_rank = l0_rank.copy()
            l0_rank.insert(0, 'partition_mode', mode.upper())
            all_l0_rank.append(l0_rank)
        if pair_rank_w is not None:
            pair_rank_w = pair_rank_w.copy()
            pair_rank_w.insert(0, 'partition_mode', mode.upper())
            all_pair_rank_w.append(pair_rank_w)
        if pair_rank_m is not None:
            pair_rank_m = pair_rank_m.copy()
            pair_rank_m.insert(0, 'partition_mode', mode.upper())
            all_pair_rank_m.append(pair_rank_m)

    if not all_results:
        raise FileNotFoundError('No local_agents_{mode}.joblib files were found in models/.')

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(out_dir / 'best_pair_vs_baselines_all.csv', index=False)

    for mode in res_df['partition_mode'].unique():
        res_df[res_df['partition_mode'] == mode].to_csv(out_dir / f'best_pair_vs_baselines_{mode.lower()}.csv', index=False)

    if all_l0_rank:
        pd.concat(all_l0_rank, ignore_index=True).to_csv(out_dir / 'best_pair_l0_validation_rank.csv', index=False)
    if all_pair_rank_w:
        pd.concat(all_pair_rank_w, ignore_index=True).to_csv(out_dir / 'best_pair_pair_rank_weighted.csv', index=False)
    if all_pair_rank_m:
        pd.concat(all_pair_rank_m, ignore_index=True).to_csv(out_dir / 'best_pair_pair_rank_mean.csv', index=False)

    summary = []
    for mode, grp in res_df.groupby('partition_mode'):
        row_l0 = grp[grp['method'] == 'Best_L0'].iloc[0]
        row_pw = grp[grp['method'] == 'Best_Pair_L2_weighted'].iloc[0]
        row_pm = grp[grp['method'] == 'Best_Pair_L2_mean'].iloc[0]
        summary.append({
            'partition_mode': mode,
            'best_l0_tpr': row_l0['test_tpr_at_fpr1'],
            'best_l0_pr': row_l0['test_pr_auc'],
            'best_l0_note': row_l0['selection_note'],
            'best_pair_weighted_tpr': row_pw['test_tpr_at_fpr1'],
            'best_pair_weighted_pr': row_pw['test_pr_auc'],
            'best_pair_weighted_note': row_pw['selection_note'],
            'best_pair_mean_tpr': row_pm['test_tpr_at_fpr1'],
            'best_pair_mean_pr': row_pm['test_pr_auc'],
            'best_pair_mean_note': row_pm['selection_note'],
            'gain_weighted_vs_l0': row_pw['test_tpr_at_fpr1'] - row_l0['test_tpr_at_fpr1'],
            'gain_mean_vs_l0': row_pm['test_tpr_at_fpr1'] - row_l0['test_tpr_at_fpr1'],
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / 'best_pair_summary.csv', index=False)

    lines = ['# Best-Pair vs Baselines Summary', '']
    for _, r in summary_df.iterrows():
        lines.append(f"## {r['partition_mode']}")
        lines.append(f"- Best_L0: TPR@1%FPR={r['best_l0_tpr']:.4f}, PR-AUC={r['best_l0_pr']:.4f}, selected={r['best_l0_note']}")
        lines.append(f"- Best_Pair_L2_weighted: TPR@1%FPR={r['best_pair_weighted_tpr']:.4f}, PR-AUC={r['best_pair_weighted_pr']:.4f}, pair={r['best_pair_weighted_note']}")
        lines.append(f"- Best_Pair_L2_mean: TPR@1%FPR={r['best_pair_mean_tpr']:.4f}, PR-AUC={r['best_pair_mean_pr']:.4f}, pair={r['best_pair_mean_note']}")
        lines.append(f"- Gain(weighted vs L0): {r['gain_weighted_vs_l0']:+.4f}")
        lines.append(f"- Gain(mean vs L0): {r['gain_mean_vs_l0']:+.4f}")
        lines.append('')
    (out_dir / 'best_pair_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    save_plots(res_df, out_dir)
    print(f'[OK] wrote results to {out_dir}')


if __name__ == '__main__':
    main()
