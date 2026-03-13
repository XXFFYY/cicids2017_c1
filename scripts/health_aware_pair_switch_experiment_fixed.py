from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = 'data_processed/cicids2017_all.parquet'
MODEL_DIR = 'models'

RANDOM_SEED = 42
TARGET_FPR = 0.01
EPS = 1e-6
PARTITIONS = ['aap', 'supervised']

OUTPUT_DIR = 'results_health_aware_switch_v2'
WINDOW_SIZE = 10000
TAU_PAIR = 0.55
TAU_SINGLE = 0.45
TOP_BACKUP_K = 3
TAU_DISAGREE = 0.15
TAU_MARGIN = 0.18


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


def compute_agent_health(ref_mat: np.ndarray, cur_mat: np.ndarray) -> np.ndarray:
    ref_mean = ref_mat.mean(axis=0)
    cur_mean = cur_mat.mean(axis=0)
    ref_std = ref_mat.std(axis=0) + EPS
    cur_std = cur_mat.std(axis=0)
    shift = np.abs(cur_mean - ref_mean) / ref_std + 0.5 * np.abs(cur_std - ref_std) / ref_std
    health = np.exp(-shift)
    return np.clip(health, 0.05, 1.0)


def load_day_splits(df: pd.DataFrame):
    day = df['day'].astype(str).str.lower()
    train_days = {'tuesday', 'wednesday', 'thursday'}
    train_df = df[day.isin(train_days)].copy()
    test_df = df[day == 'friday'].copy()
    monday_df = df[day == 'monday'].copy()
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


def rows_to_matrix(rows: List[Dict], key: str) -> Tuple[List[str], np.ndarray]:
    names = [r['agent'] for r in rows]
    mat = np.column_stack([r[key] for r in rows])
    return names, mat


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


def rank_pairs(rows: List[Dict], y_va: np.ndarray, weighted: bool = True) -> pd.DataFrame:
    out = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            fused = fuse_pair(rows[i], rows[j], y_va, weighted=weighted)
            thr = threshold_from_calib(fused['p_ca'])
            val_tpr = tpr_at_threshold(y_va, fused['p_va'], thr)
            val_pr = float(average_precision_score(y_va, fused['p_va']))
            val_roc = float(roc_auc_score(y_va, fused['p_va']))
            out.append({
                'agent_a': fused['agents'][0],
                'agent_b': fused['agents'][1],
                'val_proxy_tpr': val_tpr,
                'val_pr_auc': val_pr,
                'val_roc_auc': val_roc,
                'selection_note': fused['note'],
            })
    return pd.DataFrame(out).sort_values(['val_proxy_tpr', 'val_pr_auc', 'val_roc_auc'], ascending=False).reset_index(drop=True)


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


def choose_backup_pair(pair_rank: pd.DataFrame, bad_agents: set[str]) -> Tuple[str, str] | None:
    for _, row in pair_rank.iterrows():
        pair = (row['agent_a'], row['agent_b'])
        if pair[0] not in bad_agents and pair[1] not in bad_agents:
            return pair
    return None


def health_aware_route(
    health: np.ndarray,
    agent_names: List[str],
    primary_pair: Tuple[str, str],
    backup_pairs: List[Tuple[str, str]],
    best_single_rank: pd.DataFrame,
    cur_mat: np.ndarray,
    tau_pair: float,
    tau_single: float,
    tau_disagree: float,
    tau_margin: float,
) -> Tuple[str, Dict]:
    name_to_idx = {n: i for i, n in enumerate(agent_names)}

    pair_health = {
        pair: float(np.mean([health[name_to_idx[pair[0]]], health[name_to_idx[pair[1]]]]))
        for pair in [primary_pair] + backup_pairs
    }

    ia = name_to_idx[primary_pair[0]]
    ib = name_to_idx[primary_pair[1]]
    primary_h = pair_health[primary_pair]

    disagree = float(np.mean(np.abs(cur_mat[:, ia] - cur_mat[:, ib])))
    margin = float(np.mean(np.maximum(np.abs(cur_mat[:, ia] - 0.5), np.abs(cur_mat[:, ib] - 0.5))))

    anomaly = (min(health[ia], health[ib]) < tau_pair) and (disagree > tau_disagree) and (margin < tau_margin)

    if not anomaly:
        return 'primary_pair', {
            'pair': primary_pair,
            'pair_health': primary_h,
            'primary_disagree': disagree,
            'primary_margin': margin,
        }

    best_agent = None
    best_agent_h = -1.0
    for _, row in best_single_rank.iterrows():
        agent = row['agent']
        h = float(health[name_to_idx[agent]])
        if h >= tau_single and h > best_agent_h:
            best_agent = agent
            best_agent_h = h

    if best_agent is None:
        best_agent = best_single_rank.iloc[0]['agent']
        best_agent_h = float(health[name_to_idx[best_agent]])

    healthy_backups = []
    for pair in backup_pairs:
        pa, pb = pair
        ph = min(float(health[name_to_idx[pa]]), float(health[name_to_idx[pb]]))
        if ph >= max(tau_pair, best_agent_h):
            healthy_backups.append((pair, ph))

    if healthy_backups:
        best_backup = max(healthy_backups, key=lambda x: x[1])[0]
        return 'backup_pair', {
            'pair': best_backup,
            'pair_health': pair_health[best_backup],
            'primary_disagree': disagree,
            'primary_margin': margin,
        }

    return 'single_fallback', {
        'agent': best_agent,
        'agent_health': best_agent_h,
        'primary_disagree': disagree,
        'primary_margin': margin,
    }


def run_partition(mode: str):
    data_path = Path(DATA_PATH)
    model_path = Path(MODEL_DIR) / f'local_agents_{mode}.joblib'
    if not data_path.exists() or not model_path.exists():
        raise FileNotFoundError(f'Missing required file for mode={mode}: {data_path} / {model_path}')

    df = pd.read_parquet(data_path)
    va_df, test_df, cal_a_df, cal_b_df = load_day_splits(df)
    y_va = va_df['y'].values
    y_te = test_df['y'].values
    y_cb = cal_b_df['y'].values

    agents = joblib.load(model_path)
    rows = build_agent_rows(agents, va_df, test_df, cal_a_df, cal_b_df)
    agent_names, va_mat = rows_to_matrix(rows, 'p_va')
    _, te_mat = rows_to_matrix(rows, 'p_te')
    _, ca_mat = rows_to_matrix(rows, 'p_ca')
    _, cb_mat = rows_to_matrix(rows, 'p_cb')
    by_name = {r['agent']: r for r in rows}

    best_single, single_rank = select_best_single(rows, y_va)
    pair_rank = rank_pairs(rows, y_va, weighted=True)
    primary_pair = (pair_rank.iloc[0]['agent_a'], pair_rank.iloc[0]['agent_b'])
    backup_pairs: List[Tuple[str, str]] = []
    for _, r in pair_rank.iloc[1:1 + TOP_BACKUP_K].iterrows():
        backup_pairs.append((r['agent_a'], r['agent_b']))

    # Static baselines
    base_results = []
    thr_single = threshold_from_calib(best_single['p_ca'])
    val_tpr_single = tpr_at_threshold(y_va, best_single['p_va'], thr_single)
    base_results.append(asdict(evaluate_scores(
        'Global_Best_L0', mode.upper(), best_single['p_te'], best_single['p_ca'], best_single['p_cb'],
        y_te, y_cb, 1.0, val_tpr_single, best_single['agent']
    )))

    pair_obj = fuse_pair(by_name[primary_pair[0]], by_name[primary_pair[1]], y_va, weighted=True)
    thr_pair = threshold_from_calib(pair_obj['p_ca'])
    val_tpr_pair = tpr_at_threshold(y_va, pair_obj['p_va'], thr_pair)
    base_results.append(asdict(evaluate_scores(
        'Global_Best_Pair_L2', mode.upper(), pair_obj['p_te'], pair_obj['p_ca'], pair_obj['p_cb'],
        y_te, y_cb, 2.0, val_tpr_pair, pair_obj['note']
    )))

        # Health-aware routing on windows (v2: hard-trigger + low-switch validation search)
    name_to_idx = {n: i for i, n in enumerate(agent_names)}
    ref_mat = ca_mat

    def apply_windows(
        target_mat: np.ndarray,
        split_name: str,
        tau_pair: float,
        tau_single: float,
        tau_disagree: float,
        tau_margin: float,
    ):
        n = target_mat.shape[0]
        out = np.zeros(n, dtype=float)
        rows_local = []

        for start in range(0, n, WINDOW_SIZE):
            end = min(start + WINDOW_SIZE, n)
            cur = target_mat[start:end]
            health = compute_agent_health(ref_mat, cur)

            route, meta = health_aware_route(
                health,
                agent_names,
                primary_pair,
                backup_pairs,
                single_rank,
                cur,
                tau_pair,
                tau_single,
                tau_disagree,
                tau_margin,
            )

            if route in {'primary_pair', 'backup_pair'}:
                a, b = meta['pair']
                pair = fuse_pair(by_name[a], by_name[b], y_va, weighted=True)
                key_map = {
                    'test': 'p_te',
                    'calib_monitor': 'p_cb',
                    'calib_threshold': 'p_ca',
                    'validation': 'p_va',
                }
                key = key_map[split_name]
                out[start:end] = pair[key][start:end]
                used_cost = 2.0
                used_note = f"{route}:{a}+{b}"
            else:
                agent = meta['agent']
                key_map = {
                    'test': 'p_te',
                    'calib_monitor': 'p_cb',
                    'calib_threshold': 'p_ca',
                    'validation': 'p_va',
                }
                key = key_map[split_name]
                out[start:end] = by_name[agent][key][start:end]
                used_cost = 1.0
                used_note = f"single:{agent}"

            rows_local.append({
                'partition_mode': mode.upper(),
                'split': split_name,
                'start': start,
                'end': end,
                'route': route,
                'note': used_note,
                'avg_health': float(np.mean(health)),
                'min_health': float(np.min(health)),
                'primary_pair_health': float(np.mean([
                    health[name_to_idx[primary_pair[0]]],
                    health[name_to_idx[primary_pair[1]]]
                ])),
                'primary_disagree': float(meta.get('primary_disagree', np.nan)),
                'primary_margin': float(meta.get('primary_margin', np.nan)),
                'used_comm_cost': used_cost,
            })

        return out, rows_local

    # 在验证集上搜索更保守的阈值
    best_cfg = None
    best_score = None

    for tau_pair in [0.55, 0.65]:
        for tau_disagree in [0.05, 0.10]:
            for tau_margin in [0.15, 0.25]:
                pred_va_tmp, rows_va_tmp = apply_windows(
                    va_mat, 'validation',
                    tau_pair, TAU_SINGLE, tau_disagree, tau_margin
                )
                pred_ca_tmp, _ = apply_windows(
                    ca_mat, 'calib_threshold',
                    tau_pair, TAU_SINGLE, tau_disagree, tau_margin
                )
                pred_cb_tmp, _ = apply_windows(
                    cb_mat, 'calib_monitor',
                    tau_pair, TAU_SINGLE, tau_disagree, tau_margin
                )

                thr_tmp = threshold_from_calib(pred_ca_tmp)
                val_tpr_tmp = tpr_at_threshold(y_va, pred_va_tmp, thr_tmp)
                monitor_tmp = fpr_at_threshold(y_cb, pred_cb_tmp, thr_tmp)

                switch_share = 1.0 - (pd.DataFrame(rows_va_tmp)['route'].eq('primary_pair').mean())

                score = val_tpr_tmp - 0.35 * abs(monitor_tmp - TARGET_FPR) - 0.08 * switch_share
                key = (score, val_tpr_tmp, -abs(monitor_tmp - TARGET_FPR), -switch_share)

                if best_score is None or key > best_score:
                    best_score = key
                    best_cfg = (tau_pair, TAU_SINGLE, tau_disagree, tau_margin)

    tau_pair_star, tau_single_star, tau_disagree_star, tau_margin_star = best_cfg

    pred_te, route_rows_te = apply_windows(
        te_mat, 'test',
        tau_pair_star, tau_single_star, tau_disagree_star, tau_margin_star
    )
    pred_cb, route_rows_cb = apply_windows(
        cb_mat, 'calib_monitor',
        tau_pair_star, tau_single_star, tau_disagree_star, tau_margin_star
    )
    pred_ca, route_rows_ca = apply_windows(
        ca_mat, 'calib_threshold',
        tau_pair_star, tau_single_star, tau_disagree_star, tau_margin_star
    )

    route_rows = route_rows_te + route_rows_cb + route_rows_ca
    avg_cost = float(pd.DataFrame(route_rows_te)['used_comm_cost'].mean())

    robust_res = asdict(evaluate_scores(
        'HealthAware_PairSwitch_B_v2',
        mode.upper(),
        pred_te,
        pred_ca,
        pred_cb,
        y_te,
        y_cb,
        avg_cost,
        val_tpr_pair,
        f'primary={primary_pair}; backups={backup_pairs}; '
        f'tau_pair={tau_pair_star}; tau_single={tau_single_star}; '
        f'tau_disagree={tau_disagree_star}; tau_margin={tau_margin_star}'
    ))

    return pd.DataFrame(base_results + [robust_res]), pd.DataFrame(route_rows), pair_rank, single_rank



def _df_to_md_table(df: pd.DataFrame) -> str:
    """Render a simple markdown table without requiring tabulate."""
    if df is None or df.empty:
        return "_(empty)_"
    cols = list(df.columns)
    header = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "".join([header, sep] + rows)


def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_routes = []
    summaries = []
    for mode in PARTITIONS:
        df_res, df_routes, pair_rank, single_rank = run_partition(mode)
        df_res.to_csv(out_dir / f'health_aware_v2_results_{mode}.csv', index=False)
        df_routes.to_csv(out_dir / f'health_aware_v2_routes_{mode}.csv', index=False)
        pair_rank.to_csv(out_dir / f'health_aware_v2_pair_rank_{mode}.csv', index=False)
        single_rank.to_csv(out_dir / f'health_aware_v2_single_rank_{mode}.csv', index=False)
        all_results.append(df_res)
        all_routes.append(df_routes)

        route_test = df_routes[df_routes['split'] == 'test']
        route_counts = route_test['route'].value_counts(normalize=True).to_dict()
        summaries.append({
            'partition_mode': mode.upper(),
            'share_primary_pair': route_counts.get('primary_pair', 0.0),
            'share_backup_pair': route_counts.get('backup_pair', 0.0),
            'share_single_fallback': route_counts.get('single_fallback', 0.0),
            'avg_comm_cost_test': float(route_test['used_comm_cost'].mean()) if len(route_test) else np.nan,
        })

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(out_dir / 'health_aware_v2_results_all.csv', index=False)
    pd.concat(all_routes, ignore_index=True).to_csv(out_dir / 'health_aware_v2_routes_all.csv', index=False)
    pd.DataFrame(summaries).to_csv(out_dir / 'health_aware_v2_route_summary.csv', index=False)

    md = []
    md.append('# Health-aware routing (option B)\n')
    md.append('This experiment treats agent health as a **routing signal** rather than directly rewriting the classifier score.\n')
    md.append('- If the primary pair is healthy enough, keep the primary pair.\n')
    md.append('- Else, switch to a healthy backup pair.\n')
    md.append('- If no healthy pair exists, fall back to the best healthy single agent.\n')
    md.append('\n## Route summary\n')
    md.append(_df_to_md_table(pd.DataFrame(summaries)))
    md.append('\n\n## Main results\n')
    md.append(_df_to_md_table(all_df))
    (out_dir / 'health_aware_v2_summary.md').write_text('\n'.join(md), encoding='utf-8')
    print(f'[OK] wrote results to {out_dir}')


if __name__ == '__main__':
    main()
