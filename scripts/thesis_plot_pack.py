from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT =  Path(__file__).parent.parent
OUT = ROOT / 'thesis_plot_outputs'
OUT.mkdir(parents=True, exist_ok=True)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _annotate_points(ax, xs, ys, labels, dx=0.15, dy=0.005, fontsize=8):
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(str(label), (x, y), xytext=(x + dx, y + dy), textcoords='data', fontsize=fontsize)


def _best_full_rows(df: pd.DataFrame) -> pd.DataFrame:
    full = df[df['method'].str.startswith('Global_Full_L2')].copy()
    idx = full.groupby('partition_mode')['tpr_at_fpr1'].idxmax()
    out = full.loc[idx].copy().sort_values('partition_mode')
    out['group'] = 'Best_Full_L2'
    return out


def plot_stage1_non_monotonic(baseline: pd.DataFrame) -> None:
    df = baseline.copy()
    df = df.rename(columns={'comm_cost_units': 'comm_cost', 'test_tpr_at_fpr1': 'tpr_at_fpr1', 'test_pr_auc': 'pr_auc'})
    valid = df[df['monitor_fpr'] <= 0.02].copy()
    valid['label'] = valid['method']
    valid = valid.sort_values('comm_cost')

    plt.figure(figsize=(9, 5.5))
    plt.plot(valid['comm_cost'], valid['tpr_at_fpr1'], marker='o')
    _annotate_points(plt.gca(), valid['comm_cost'], valid['tpr_at_fpr1'], valid['label'], dx=0.7, dy=0.003)
    plt.xlabel('Communication cost')
    plt.ylabel('TPR @ 1% FPR')
    plt.title('Stage 1: communication increase does not guarantee higher detection')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'fig01_stage1_non_monotonic_tpr_vs_comm.png', dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.plot(valid['comm_cost'], valid['pr_auc'], marker='o')
    _annotate_points(plt.gca(), valid['comm_cost'], valid['pr_auc'], valid['label'], dx=0.7, dy=0.003)
    plt.xlabel('Communication cost')
    plt.ylabel('PR-AUC')
    plt.title('Stage 1: communication increase does not guarantee higher PR-AUC')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'fig02_stage1_non_monotonic_pr_vs_comm.png', dpi=200)
    plt.close()


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

    # grouped bars for TPR
    partitions = order
    groups = ['Best_L0', 'Best_Pair_L2', 'Best_Full_L2']
    x = np.arange(len(partitions))
    width = 0.24
    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    for i, g in enumerate(groups):
        vals = []
        for p in partitions:
            row = compare[(compare['partition_mode'] == p) & (compare['group'] == g)]
            vals.append(float(row['tpr_at_fpr1'].iloc[0]))
        ax.bar(x + (i - 1) * width, vals, width=width, label=g)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in partitions])
    ax.set_ylabel('TPR @ 1% FPR')
    ax.set_title('Stage 2: low-communication baseline selection under each partition')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'fig03_stage2_partition_tpr_bars.png', dpi=200)
    plt.close()

    # scatter on cost-vs-TPR with labels
    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    for p in partitions:
        sub = compare[compare['partition_mode'] == p].sort_values('comm_cost')
        ax.plot(sub['comm_cost'], sub['tpr_at_fpr1'], marker='o', label=p.upper())
        for _, r in sub.iterrows():
            ax.annotate(f"{p[:3]}-{r['group']}", (r['comm_cost'], r['tpr_at_fpr1']), fontsize=8,
                        xytext=(r['comm_cost'] + 0.1, r['tpr_at_fpr1'] + 0.004))
    ax.set_xlabel('Communication cost')
    ax.set_ylabel('TPR @ 1% FPR')
    ax.set_title('Stage 2: promising baselines on the communication-performance tradeoff')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'fig04_stage2_partition_tradeoff.png', dpi=200)
    plt.close()

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

    plt.figure(figsize=(8.5, 5.5))
    x = np.arange(len(order))
    width = 0.35
    ax = plt.gca()
    ax.bar(x - width / 2, gain['pair_gain_vs_single'], width=width, label='Best Pair - Best L0')
    ax.bar(x + width / 2, gain['pair_gain_vs_full'], width=width, label='Best Pair - Best Full L2')
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in order])
    ax.set_ylabel('Gain in TPR @ 1% FPR')
    ax.set_title('Stage 3: does pair selection reduce communication and improve performance?')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'fig05_stage3_pair_gain.png', dpi=200)
    plt.close()

    # AAP optimization path: best single -> best pair -> best full L2 family
    aap = fair[fair['partition_mode'] == 'aap'].copy()
    methods = [
        'Global_Best_L0',
        'Global_Best_Pair_L2_mean',
        'Global_Full_L2_mean',
        'Global_Full_L2_trimmed',
        'Global_Full_L2_attention_robust',
        'Global_Full_L2_stacking_lr',
    ]
    aap = aap[aap['method'].isin(methods)].copy()
    label_map = {
        'Global_Best_L0': 'Best L0',
        'Global_Best_Pair_L2_mean': 'Best Pair',
        'Global_Full_L2_mean': 'Full-L2 mean',
        'Global_Full_L2_trimmed': 'Full-L2 trimmed',
        'Global_Full_L2_attention_robust': 'Full-L2 robust',
        'Global_Full_L2_stacking_lr': 'Full-L2 stacking',
    }
    aap['label'] = aap['method'].map(label_map)
    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()
    ax.scatter(aap['comm_cost'], aap['tpr_at_fpr1'])
    for _, r in aap.iterrows():
        ax.annotate(r['label'], (r['comm_cost'], r['tpr_at_fpr1']), fontsize=9,
                    xytext=(r['comm_cost'] + 0.05, r['tpr_at_fpr1'] + 0.004))
    ax.set_xlabel('Communication cost')
    ax.set_ylabel('TPR @ 1% FPR')
    ax.set_title('Stage 3 focus: AAP collaborative baseline optimized from Full-L2 to Best-Pair-L2')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / 'fig06_stage3_aap_optimization_path.png', dpi=200)
    plt.close()

    return gain


def plot_appendix_robustness(robust: pd.DataFrame) -> None:
    if robust is None or robust.empty:
        return
    keep = robust[robust['scenario'].isin(['friday_halfswap', 'feature_dropout_30', 'score_noise_003'])].copy()
    if keep.empty:
        return
    objects = ['Global_Best_L0', 'Global_Best_Pair_L2']
    modes = ['aap', 'supervised']
    scenarios = ['friday_halfswap', 'feature_dropout_30', 'score_noise_003']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, mode in zip(axes, modes):
        sub = keep[keep['mode'] == mode].copy()
        x = np.arange(len(scenarios))
        width = 0.35
        for i, obj in enumerate(objects):
            vals = []
            for sc in scenarios:
                rows = sub[(sub['object'] == obj) & (sub['scenario'] == sc)]
                vals.append(float(rows['delta_tpr'].mean()) if len(rows) else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width, label=obj)
        ax.axhline(0.0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(['Fri half-swap', 'Feature drop 30%', 'Score noise'])
        ax.set_title(mode.upper())
        ax.grid(True, axis='y', alpha=0.3)
    axes[0].set_ylabel('Delta TPR @ 1% FPR')
    axes[1].legend()
    plt.suptitle('Appendix: robustness deltas for single vs pair under AAP and supervised partitions')
    plt.tight_layout()
    plt.savefig(OUT / 'figA1_appendix_robustness_delta_tpr.png', dpi=200)
    plt.close()


def write_summary_md(compare: pd.DataFrame, gain: pd.DataFrame) -> None:
    # best deployment and best collaboration
    single_rows = compare[compare['group'] == 'Best_L0'].copy()
    best_single = single_rows.loc[single_rows['tpr_at_fpr1'].idxmax()]
    pair_rows = compare[compare['group'] == 'Best_Pair_L2'].copy()
    best_pair = pair_rows.loc[pair_rows['tpr_at_fpr1'].idxmax()]
    most_gain = gain.loc[gain['pair_gain_vs_single'].idxmax()]

    text = f"""# Thesis plot plan based on current results

## Recommended mainline
1. **Stage 1 — communication is not monotonic with performance**
   - Use `fig01_stage1_non_monotonic_tpr_vs_comm.png`
   - Optional companion: `fig02_stage1_non_monotonic_pr_vs_comm.png`

2. **Stage 2 — identify promising low-communication baselines**
   - Use `fig03_stage2_partition_tpr_bars.png`
   - Use `fig04_stage2_partition_tradeoff.png`
   - Current strongest single-agent deployment point: **{best_single['partition_mode'].upper()} + Best_L0**
     - TPR@1%FPR = **{best_single['tpr_at_fpr1']:.4f}** at communication cost **{int(best_single['comm_cost'])}**

3. **Stage 3 — optimize the collaborative baseline**
   - Use `fig05_stage3_pair_gain.png`
   - Use `fig06_stage3_aap_optimization_path.png`
   - Largest pair gain over single appears under **{most_gain['partition_mode'].upper()}**
     - Pair gain vs single = **{most_gain['pair_gain_vs_single']:+.4f}**
     - Pair gain vs best full-L2 = **{most_gain['pair_gain_vs_full']:+.4f}**
   - Current strongest pair point: **{best_pair['partition_mode'].upper()} + Best_Pair_L2**
     - TPR@1%FPR = **{best_pair['tpr_at_fpr1']:.4f}** at communication cost **{int(best_pair['comm_cost'])}**

## Interpretation aligned with your current results
- If you want the cleanest **deployment baseline**, write the story around **supervised + Best_L0**.
- If you want the clearest **collaborative optimization story**, write the story around **AAP + Best_Pair_L2**, because it improves over the best full-L2 family while using lower communication.
- Therefore, your paper can explicitly separate:
  - **single-agent optimum**
  - **pair-collaboration optimum**

## Appendix suggestion
- If you keep the robustness section, use `figA1_appendix_robustness_delta_tpr.png` as a compact appendix figure.
"""
    (OUT / 'plot_design_summary.md').write_text(text, encoding='utf-8')


def main() -> None:
    baseline = _safe_read_csv(ROOT / 'results' / 'baseline_plot_table.csv')
    fair = _safe_read_csv(ROOT / 'results' / 'fair_compare_all.csv')
    robust = _safe_read_csv(ROOT / 'results' / 'robustness_summary.csv')

    if baseline is None:
        raise FileNotFoundError('Missing baseline_plot_table.csv')
    if fair is None:
        raise FileNotFoundError('Missing fair_compare_all.csv')

    plot_stage1_non_monotonic(baseline)
    compare = plot_stage2_partition_summary(fair)
    gain = plot_stage3_optimization(fair)
    plot_appendix_robustness(robust)
    write_summary_md(compare, gain)
    print(f'Wrote figures and summary to: {OUT}')


if __name__ == '__main__':
    main()
