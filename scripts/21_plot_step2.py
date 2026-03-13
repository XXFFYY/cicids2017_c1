from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from low_comm_ids import ExperimentConfig
from low_comm_ids.plotting import save_gain_bar_plot, save_grouped_bar_plot


if __name__ == "__main__":
    cfg = ExperimentConfig(project_root=ROOT)
    metrics_path = cfg.step2_results_path / "l2_optimization_metrics.csv"
    gains_path = cfg.step2_results_path / "aap_gain_by_fusion.csv"
    within_path = cfg.step2_results_path / "l2_gain_breakdown.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run scripts/20_run_step2_l2_optimization.py first.")

    df = pd.read_csv(metrics_path)
    save_grouped_bar_plot(
        df=df,
        index_col="method",
        group_col="partition_mode",
        value_col="test_pr_auc",
        title="Step 2: L2 PR-AUC (Random vs AAP)",
        xlabel="L2 fusion method",
        ylabel="PR-AUC on Friday",
        out_path=cfg.step2_results_path / "fig_step2_pr_auc_compare.png",
        order=["L2_mean", "L2_trimmed_mean", "L2_stacking_lr", "L2_attention_ap", "L2_attention_robust"],
    )
    save_grouped_bar_plot(
        df=df,
        index_col="method",
        group_col="partition_mode",
        value_col="test_tpr_at_fpr1",
        title="Step 2: L2 TPR@FPR=1% (Random vs AAP)",
        xlabel="L2 fusion method",
        ylabel="TPR on Friday at 1% FPR",
        out_path=cfg.step2_results_path / "fig_step2_tpr1_compare.png",
        order=["L2_mean", "L2_trimmed_mean", "L2_stacking_lr", "L2_attention_ap", "L2_attention_robust"],
    )

    if gains_path.exists():
        gains_df = pd.read_csv(gains_path)
        save_gain_bar_plot(
            df=gains_df,
            x_col="method",
            y_col="tpr_gain_aap_minus_random",
            title="Step 2: TPR Gain from AAP over Random",
            xlabel="L2 fusion method",
            ylabel="TPR gain at 1% FPR",
            out_path=cfg.step2_results_path / "fig_step2_tpr_gain_aap_minus_random.png",
        )

    if within_path.exists():
        within_df = pd.read_csv(within_path)
        aap_df = within_df[within_df["partition_mode"] == "AAP"].copy()
        save_gain_bar_plot(
            df=aap_df,
            x_col="method",
            y_col="tpr_gain_vs_same_partition_l2_mean",
            title="Step 2: Within-AAP TPR Gain over Plain L2-Mean",
            xlabel="L2 fusion method",
            ylabel="TPR gain at 1% FPR",
            out_path=cfg.step2_results_path / "fig_step2_tpr_gain_within_aap.png",
        )

    print("Saved step 2 figures to:", cfg.step2_results_path)
