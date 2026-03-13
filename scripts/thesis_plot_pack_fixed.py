from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT =  Path(__file__).parent.parent
OUT = ROOT / 'thesis_plot_outputs_fixed'
OUT.mkdir(parents=True, exist_ok=True)


def _safe_read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _apply_label_offsets(ax, df, xcol, ycol, labelcol, offsets, fontsize=8):
    for _, r in df.iterrows():
        label = str(r[labelcol])
        dx, dy = offsets.get(label, (0.5, 0.004))
        ax.annotate(
            label,
            (r[xcol], r[ycol]),
            xytext=(r[xcol] + dx, r[ycol] + dy),
            textcoords='data',
            fontsize=fontsize,
            arrowprops=dict(arrowstyle='-', lw=0.5, alpha=0.6),
            bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='none', alpha=0.75),
        )


def _best_full_rows(df: pd.DataFrame) -> pd.DataFrame:
    full = df[df['method'].str.startswith('Global_Full_L2')].copy()
    idx = full.groupby('partition_mode')['tpr_at_fpr1'].idxmax()
    out = full.loc[idx].copy().sort_values('partition_mode')
    out['group'] = 'Best_Full_L2'
    return out


def plot_stage1_non_monotonic(baseline: pd.DataFrame) -> None:
    df = baseline.rename(columns={
        'comm_cost_units': 'comm_cost',
        'test_tpr_at_fpr1': 'tpr_at_fpr1',
        'test_pr_auc': 'pr_auc',
    }).copy()
    valid = df[df['monitor_fpr'] <= 0.02].copy().sort_values(['comm_cost', 'tpr_at_fpr1'])
    label_map = {
        'Best_L0': 'Best-L0',
        'L2_mean': 'L2-mean',
        'L2_stacking_lr': 'L2-stack',
        'L2_trimmed_mean': 'L2-trim',
        'L3_pca_d1': 'L3-d1',
        'L3_pca_d4': 'L3-d4',
        'L3_pca_d8': 'L3-d8',
        'L3_pca_d16': 'L3-d16',
        'L3_pca_d32': 'L3-d32',
        'Centralized': 'Central',
    }
    valid['label'] = valid['method'].map(label_map).fillna(valid['method'])

    offsets_tpr = {
        'Best-L0': (0.75, 0.010),
        'L3-d1': (0.95, -0.020),
        'L2-mean': (0.95, -0.010),
        'L2-stack': (0.95, 0.002),
        'L2-trim': (0.95, 0.014),
        'L3-d4': (0.80, 0.010),
        'L3-d8': (0.80, 0.008),
        'L3-d16': (0.80, 0.010),
        'Central': (-12.0, 0.008),
        'L3-d32': (-11.0, 0.010),
    }
    offsets_pr = {
        'Best-L0': (0.75, 0.008),
        'L2-mean': (0.95, -0.018),
        'L2-stack': (0.95, -0.008),
        'L2-trim': (0.95, 0.006),
        'L3-d1': (0.95, 0.018),
        'L3-d4': (0.80, 0.008),
        'L3-d8': (0.80, 0.008),
        'L3-d16': (0.80, 0.008),
        'Central': (-10.0, -0.020),
        'L3-d32': (-12.0, 0.008),
    }

    fig, ax = plt.subplots(figsize=(9.4, 5.7))
    ax.plot(valid['comm_cost'], valid['tpr_at_fpr1'], marker='o')
    _apply_label_offsets(ax, valid, 'comm_cost', 'tpr_at_fpr1', 'label', offsets_tpr, fontsize=8)
    ax.set_xlabel('Communication cost')
    ax.set_ylabel('TPR @ 1% FPR')
    ax.set_title('Stage 1: higher communication does not guarantee higher TPR')
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.08, y=0.12)
    fig.tight_layout()
    fig.savefig(OUT / 'fig01_stage1_non_monotonic_tpr_vs_comm_fixed.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.7))
    ax.plot(valid['comm_cost'], valid['pr_auc'], marker='o')
    _apply_label_offsets(ax, valid, 'comm_cost', 'pr_auc', 'label', offsets_pr, fontsize=8)
    ax.set_xlabel('Communication cost')
    ax.set_ylabel('PR-AUC')
    ax.set_title('Stage 1: higher communication does not guarantee higher PR-AUC')
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.08, y=0.12)
    fig.tight_layout()
    fig.savefig(OUT / 'fig02_stage1_non_monotonic_pr_vs_comm_fixed.png', dpi=220)
    plt.close(fig)


def plot_stage2_partition_summary(fair: pd.DataFrame) -> pd.DataFrame:
    single = fair[fair['method'] == 'Global_Best_L0'].copy()
    pair = fair[fair['method'] == 'Global_Best_Pair_L2_mean'].copy()
    pair['group'] = 'Best_Pair_L2'
    single['group'] = 'Best_L0'
    full = _best_full_rows(fair)
    compare = pd.concat([
        single[['partition_mode', 'group', 'comm_cost', 'tpr_at_fpr1', 'pr_auc', 'note']],
        pair[['partition_mode', 'group', 'comm_cost', 'tpr_at_fpr1', 'pr_auc', 'note']],
        full[['partition_mode', 'group', 'comm_cost', 'tpr_at_fpr1', 'pr_auc', 'note']],
    ], ignore_index=True)
    order = ['random', 'aap', 'supervised']
    compare['partition_mode'] = pd.Categorical(compare['partition_mode'], categories=order, ordered=True)
    compare = compare.sort_values(['partition_mode', 'group'])
    compare.to_csv(OUT / 'stage2_partition_summary_table.csv', index=False)
    return compare


def plot_stage3_optimization(fair: pd.DataFrame) -> pd.DataFrame:
    order = ['random', 'aap', 'supervised']
    rows = []
    for p in order:
        sub = fair[fair['partition_mode'] == p].copy()
        single = sub[sub['method'] == 'Global_Best_L0'].iloc[0]
        pair = sub[sub['method'] == 'Global_Best_Pair_L2_mean'].iloc[0]
        full = _best_full_rows(sub).iloc[0]
        rows.append({
            'partition_mode': p,
            'single_tpr': single['tpr_at_fpr1'],
            'pair_tpr': pair['tpr_at_fpr1'],
            'full_tpr': full['tpr_at_fpr1'],
            'pair_gain_vs_single': pair['tpr_at_fpr1'] - single['tpr_at_fpr1'],
            'pair_gain_vs_full': pair['tpr_at_fpr1'] - full['tpr_at_fpr1'],
            'single_comm': single['comm_cost'],
            'pair_comm': pair['comm_cost'],
            'full_comm': full['comm_cost'],
            'pair_note': pair['note'],
            'full_note': full['note'],
        })
    gain = pd.DataFrame(rows)
    gain.to_csv(OUT / 'stage3_pair_gain_table.csv', index=False)

    aap = fair[(fair['partition_mode'] == 'aap') & (fair['method'].isin([
        'Global_Best_L0',
        'Global_Best_Pair_L2_mean',
        'Global_Full_L2_mean',
        'Global_Full_L2_trimmed',
        'Global_Full_L2_attention_robust',
        'Global_Full_L2_stacking_lr',
    ]))].copy()
    label_map = {
        'Global_Best_L0': 'Best L0',
        'Global_Best_Pair_L2_mean': 'Best pair',
        'Global_Full_L2_mean': 'Full mean',
        'Global_Full_L2_trimmed': 'Full trim',
        'Global_Full_L2_attention_robust': 'Full robust',
        'Global_Full_L2_stacking_lr': 'Full stack',
    }
    aap['label'] = aap['method'].map(label_map)
    offsets = {
        'Best L0': (0.10, 0.010),
        'Best pair': (0.10, 0.008),
        'Full mean': (0.12, -0.012),
        'Full stack': (0.12, 0.000),
        'Full robust': (0.12, 0.012),
        'Full trim': (0.12, 0.024),
    }
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    ax.scatter(aap['comm_cost'], aap['tpr_at_fpr1'])
    _apply_label_offsets(ax, aap, 'comm_cost', 'tpr_at_fpr1', 'label', offsets, fontsize=9)
    ax.set_xlabel('Communication cost')
    ax.set_ylabel('TPR @ 1% FPR')
    ax.set_title('Stage 3 focus: AAP path from single-agent to best pair and full L2')
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.10, y=0.18)
    fig.tight_layout()
    fig.savefig(OUT / 'fig06_stage3_aap_optimization_path_fixed.png', dpi=220)
    plt.close(fig)
    return gain


def main():
    baseline = _safe_read_csv(ROOT / 'results' / 'baseline_plot_table.csv')
    fair = _safe_read_csv(ROOT / 'results' / 'fair_compare_all.csv')
    if baseline is not None:
        plot_stage1_non_monotonic(baseline)
    if fair is not None:
        plot_stage2_partition_summary(fair)
        plot_stage3_optimization(fair)


if __name__ == '__main__':
    main()
