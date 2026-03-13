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
OUTPUT_DIR = 'results_agent_failure_response'
RANDOM_SEED = 42
TARGET_FPR = 0.01
PARTITIONS = ['aap', 'supervised']
ATTACKS = ['score_zero', 'score_flip', 'score_noise_003']
EPS = 1e-6


@dataclass
class EvalResult:
    partition_mode: str
    base_method: str
    compromised_agent: str
    attack_type: str
    response_strategy: str
    avg_comm_cost: float
    test_roc_auc: float
    test_pr_auc: float
    test_tpr_at_fpr1: float
    monitor_fpr: float
    decision_threshold: float
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


def evaluate_scores(partition_mode: str, base_method: str, compromised_agent: str, attack_type: str,
                    response_strategy: str, p_test: np.ndarray, p_calib_threshold: np.ndarray,
                    p_calib_monitor: np.ndarray, y_test: np.ndarray, y_calib_monitor: np.ndarray,
                    avg_comm_cost: float, selection_note: str) -> EvalResult:
    thr = threshold_from_calib(p_calib_threshold)
    return EvalResult(
        partition_mode=partition_mode.upper(),
        base_method=base_method,
        compromised_agent=compromised_agent,
        attack_type=attack_type,
        response_strategy=response_strategy,
        avg_comm_cost=float(avg_comm_cost),
        test_roc_auc=float(roc_auc_score(y_test, p_test)),
        test_pr_auc=float(average_precision_score(y_test, p_test)),
        test_tpr_at_fpr1=tpr_at_threshold(y_test, p_test, thr),
        monitor_fpr=fpr_at_threshold(y_calib_monitor, p_calib_monitor, thr),
        decision_threshold=thr,
        selection_note=selection_note,
    )


def load_day_splits(df: pd.DataFrame):
    day = df['day'].astype(str).str.lower()
    df = df.copy()
    df['day'] = day
    train_days = {'tuesday', 'wednesday', 'thursday'}
    train_df = df[df['day'].isin(train_days)].copy()
    test_df = df[df['day'] == 'friday'].copy()
    monday_df = df[df['day'] == 'monday'].copy()
    _, va_df = train_test_split(train_df, test_size=0.2, stratify=train_df['y'], random_state=RANDOM_SEED)
    cal_a_df, cal_b_df = train_test_split(monday_df, test_size=0.5, stratify=monday_df['y'], random_state=RANDOM_SEED)
    return va_df, test_df, cal_a_df, cal_b_df


def normalize_cols(cols):
    if cols is None:
        return []
    if isinstance(cols, str):
        cols = [cols]
    elif hasattr(cols, 'tolist') and not isinstance(cols, list):
        cols = cols.tolist()
    if not isinstance(cols, list):
        cols = list(cols)
    out = []
    seen = set()
    for c in cols:
        if isinstance(c, str) and c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build_agent_rows(agents: Dict, va_df: pd.DataFrame, test_df: pd.DataFrame, cal_a_df: pd.DataFrame, cal_b_df: pd.DataFrame):
    rows = []
    for name, bundle in agents.items():
        model = bundle['model']
        cols = normalize_cols(bundle['cols'])
        rows.append({
            'agent': name,
            'cols': cols,
            'p_va': model.predict_proba(va_df.loc[:, cols])[:, 1],
            'p_te': model.predict_proba(test_df.loc[:, cols])[:, 1],
            'p_ca': model.predict_proba(cal_a_df.loc[:, cols])[:, 1],
            'p_cb': model.predict_proba(cal_b_df.loc[:, cols])[:, 1],
        })
    return rows


def select_best_single(rows: List[Dict], y_va: np.ndarray):
    best = None
    best_key = None
    rank_rows = []
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
        note = f'{a["agent"]}+{b["agent"]}|w={w[0]:.3f},{w[1]:.3f}'
    else:
        w = np.array([0.5, 0.5])
        note = f'{a["agent"]}+{b["agent"]}|mean'
    return {
        'agents': (a['agent'], b['agent']),
        'weights': w,
        'note': note,
        'p_va': w[0] * a['p_va'] + w[1] * b['p_va'],
        'p_te': w[0] * a['p_te'] + w[1] * b['p_te'],
        'p_ca': w[0] * a['p_ca'] + w[1] * b['p_ca'],
        'p_cb': w[0] * a['p_cb'] + w[1] * b['p_cb'],
    }


def rank_pairs(rows: List[Dict], y_va: np.ndarray, weighted: bool = True):
    ranked = []
    best = None
    best_key = None
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            fused = fuse_pair(rows[i], rows[j], y_va, weighted=weighted)
            thr = threshold_from_calib(fused['p_ca'])
            val_tpr = tpr_at_threshold(y_va, fused['p_va'], thr)
            val_pr = float(average_precision_score(y_va, fused['p_va']))
            val_roc = float(roc_auc_score(y_va, fused['p_va']))
            rec = {
                'agent_a': fused['agents'][0],
                'agent_b': fused['agents'][1],
                'val_proxy_tpr': val_tpr,
                'val_pr_auc': val_pr,
                'val_roc_auc': val_roc,
                'selection_note': fused['note'],
            }
            ranked.append(rec)
            key = (val_tpr, val_pr, val_roc)
            if best_key is None or key > best_key:
                best_key = key
                best = fused
    rank_df = pd.DataFrame(ranked).sort_values(['val_proxy_tpr', 'val_pr_auc', 'val_roc_auc'], ascending=False).reset_index(drop=True)
    return best, rank_df


def apply_attack(p: np.ndarray, attack: str, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if attack == 'score_zero':
        return np.zeros_like(p)
    if attack == 'score_flip':
        return 1.0 - p
    if attack == 'score_noise_003':
        return np.clip(p + rng.normal(0.0, 0.03, size=p.shape), 0.0, 1.0)
    raise ValueError(f'unknown attack: {attack}')


def attacked_row(row: Dict, attack: str) -> Dict:
    return {
        'agent': row['agent'],
        'cols': row['cols'],
        'p_va': apply_attack(row['p_va'], attack, 101),
        'p_te': apply_attack(row['p_te'], attack, 102),
        'p_ca': apply_attack(row['p_ca'], attack, 103),
        'p_cb': apply_attack(row['p_cb'], attack, 104),
    }


def merge_weighted(a: Dict, b: Dict, weights: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        'p_te': weights[0] * a['p_te'] + weights[1] * b['p_te'],
        'p_ca': weights[0] * a['p_ca'] + weights[1] * b['p_ca'],
        'p_cb': weights[0] * a['p_cb'] + weights[1] * b['p_cb'],
    }


def select_backup_pair(pair_rank_df: pd.DataFrame, compromised: str):
    df = pair_rank_df[(pair_rank_df['agent_a'] != compromised) & (pair_rank_df['agent_b'] != compromised)]
    if df.empty:
        return None
    return df.iloc[0]


def run_partition(mode: str):
    model_path = Path(MODEL_DIR) / f'local_agents_{mode}.joblib'
    if not model_path.exists():
        return None, None, None

    df = pd.read_parquet(DATA_PATH)
    va_df, test_df, cal_a_df, cal_b_df = load_day_splits(df)
    y_va = va_df['y'].values
    y_te = test_df['y'].values
    y_cb = cal_b_df['y'].values

    agents = joblib.load(model_path)
    rows = build_agent_rows(agents, va_df, test_df, cal_a_df, cal_b_df)
    by_name = {r['agent']: r for r in rows}

    best_single, single_rank = select_best_single(rows, y_va)
    best_pair, pair_rank = rank_pairs(rows, y_va, weighted=True)

    base_rows = []
    base_rows.append(evaluate_scores(mode, 'Global_Best_L0', 'none', 'clean', 'clean_single',
                                     best_single['p_te'], best_single['p_ca'], best_single['p_cb'], y_te, y_cb,
                                     1, best_single['agent']))
    base_rows.append(evaluate_scores(mode, 'Global_Best_Pair_L2', 'none', 'clean', 'clean_pair',
                                     best_pair['p_te'], best_pair['p_ca'], best_pair['p_cb'], y_te, y_cb,
                                     2, best_pair['note']))

    results = [asdict(r) for r in base_rows]
    pair_agents = list(best_pair['agents'])
    best_single_name = best_single['agent']

    for compromised in sorted(set(pair_agents + [best_single_name])):
        attacked = attacked_row(by_name[compromised], 'score_zero')  # template not used directly
        for attack in ATTACKS:
            victim = attacked_row(by_name[compromised], attack)
            local_by_name = dict(by_name)
            local_by_name[compromised] = victim

            # Strategy 1: keep compromised pair
            a_name, b_name = pair_agents
            keep_pair = merge_weighted(local_by_name[a_name], local_by_name[b_name], best_pair['weights'])
            results.append(asdict(evaluate_scores(
                mode, 'Global_Best_Pair_L2', compromised, attack, 'keep_compromised_pair',
                keep_pair['p_te'], keep_pair['p_ca'], keep_pair['p_cb'], y_te, y_cb, 2,
                best_pair['note']
            )))

            # Strategy 2: isolate bad agent and use healthy partner only if compromised is in pair
            if compromised in pair_agents:
                healthy = b_name if compromised == a_name else a_name
                hrow = local_by_name[healthy]
                results.append(asdict(evaluate_scores(
                    mode, 'Global_Best_Pair_L2', compromised, attack, 'isolate_to_healthy_partner_single',
                    hrow['p_te'], hrow['p_ca'], hrow['p_cb'], y_te, y_cb, 1, healthy
                )))

            # Strategy 3: fallback to best healthy single (exclude compromised)
            healthy_single_df = single_rank[single_rank['agent'] != compromised].reset_index(drop=True)
            if not healthy_single_df.empty:
                fallback_single_name = healthy_single_df.iloc[0]['agent']
                srow = local_by_name[fallback_single_name]
                results.append(asdict(evaluate_scores(
                    mode, 'Global_Best_L0', compromised, attack, 'fallback_best_healthy_single',
                    srow['p_te'], srow['p_ca'], srow['p_cb'], y_te, y_cb, 1, fallback_single_name
                )))

            # Strategy 4: switch to backup pair excluding compromised
            backup = select_backup_pair(pair_rank, compromised)
            if backup is not None:
                aa = local_by_name[backup['agent_a']]
                bb = local_by_name[backup['agent_b']]
                backup_fused = fuse_pair(aa, bb, y_va, weighted=True)
                results.append(asdict(evaluate_scores(
                    mode, 'Global_Best_Pair_L2', compromised, attack, 'switch_backup_pair',
                    backup_fused['p_te'], backup_fused['p_ca'], backup_fused['p_cb'], y_te, y_cb, 2,
                    backup_fused['note']
                )))

    res_df = pd.DataFrame(results)
    base_df = res_df[res_df['attack_type'] == 'clean'][['partition_mode', 'base_method', 'response_strategy', 'test_tpr_at_fpr1', 'monitor_fpr']].copy()
    base_df = base_df.rename(columns={
        'response_strategy': 'base_response_strategy',
        'test_tpr_at_fpr1': 'base_tpr_at_fpr1',
        'monitor_fpr': 'base_monitor_fpr',
    })
    attacked_df = res_df[res_df['attack_type'] != 'clean'].copy()
    attacked_df = attacked_df.merge(base_df, on=['partition_mode', 'base_method'], how='left')
    attacked_df['delta_tpr'] = attacked_df['test_tpr_at_fpr1'] - attacked_df['base_tpr_at_fpr1']
    attacked_df['delta_monitor_fpr'] = attacked_df['monitor_fpr'] - attacked_df['base_monitor_fpr']

    summary = attacked_df.groupby(['partition_mode', 'base_method', 'attack_type', 'response_strategy'], as_index=False).agg(
        avg_tpr=('test_tpr_at_fpr1', 'mean'),
        avg_delta_tpr=('delta_tpr', 'mean'),
        avg_monitor_fpr=('monitor_fpr', 'mean'),
        avg_delta_monitor_fpr=('delta_monitor_fpr', 'mean'),
        avg_comm_cost=('avg_comm_cost', 'mean'),
        n_cases=('compromised_agent', 'count'),
    )
    return res_df, attacked_df, summary, pair_rank, single_rank


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    all_raw = []
    all_delta = []
    all_summary = []

    for mode in PARTITIONS:
        result = run_partition(mode)
        if result[0] is None:
            continue
        raw_df, delta_df, summary_df, pair_rank, single_rank = result
        raw_df.to_csv(out / f'agent_failure_response_{mode}.csv', index=False)
        delta_df.to_csv(out / f'agent_failure_response_delta_{mode}.csv', index=False)
        summary_df.to_csv(out / f'agent_failure_response_summary_{mode}.csv', index=False)
        pair_rank.to_csv(out / f'agent_failure_pair_rank_{mode}.csv', index=False)
        single_rank.to_csv(out / f'agent_failure_single_rank_{mode}.csv', index=False)
        all_raw.append(raw_df)
        all_delta.append(delta_df)
        all_summary.append(summary_df)

    if all_raw:
        raw = pd.concat(all_raw, ignore_index=True)
        delta = pd.concat(all_delta, ignore_index=True)
        summary = pd.concat(all_summary, ignore_index=True)
        raw.to_csv(out / 'agent_failure_response_all.csv', index=False)
        delta.to_csv(out / 'agent_failure_response_delta_all.csv', index=False)
        summary.to_csv(out / 'agent_failure_response_summary_all.csv', index=False)

        key = ['partition_mode', 'base_method', 'attack_type', 'avg_delta_tpr']
        md_lines = ['# Agent Failure Response Summary', '']
        for mode in sorted(summary['partition_mode'].unique()):
            md_lines.append(f'## {mode}')
            sub = summary[summary['partition_mode'] == mode].copy()
            for base_method in sub['base_method'].unique():
                md_lines.append(f'### {base_method}')
                part = sub[sub['base_method'] == base_method].sort_values(['attack_type', 'avg_delta_tpr'], ascending=[True, False])
                for attack in part['attack_type'].unique():
                    md_lines.append(f'- {attack}')
                    att = part[part['attack_type'] == attack]
                    for _, r in att.iterrows():
                        md_lines.append(
                            f'  - {r.response_strategy}: avg_tpr={r.avg_tpr:.4f}, delta_tpr={r.avg_delta_tpr:+.4f}, '
                            f'avg_monitor_fpr={r.avg_monitor_fpr:.4f}, cost={r.avg_comm_cost:.1f}'
                        )
                md_lines.append('')
        (out / 'agent_failure_response_summary.md').write_text('\n'.join(md_lines), encoding='utf-8')


if __name__ == '__main__':
    main()
