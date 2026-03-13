from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from low_comm_ids import ExperimentConfig
from low_comm_ids.data import build_processed_dataset
from low_comm_ids.experiment_step1 import run_step1_baselines
from low_comm_ids.experiment_step2 import run_step2_l2_optimization
from low_comm_ids.plotting import save_grouped_bar_plot, save_line_plot, save_gain_bar_plot
import pandas as pd


if __name__ == "__main__":
    cfg = ExperimentConfig(project_root=ROOT)
    if not cfg.processed_path.exists():
        build_processed_dataset(cfg)

    run_step1_baselines(cfg)
    step1_df = pd.read_csv(cfg.step1_results_path / "baseline_plot_table.csv").sort_values(["comm_cost_units", "method"])
    save_line_plot(step1_df, "comm_cost_units", "test_pr_auc", "method", "Step 1: PR-AUC vs Communication Cost", "Communication cost (scalar-equivalent units per sample)", "PR-AUC on Friday", cfg.step1_results_path / "fig_step1_pr_auc_vs_comm.png")
    save_line_plot(step1_df, "comm_cost_units", "test_tpr_at_fpr1", "method", "Step 1: TPR@FPR=1% vs Communication Cost", "Communication cost (scalar-equivalent units per sample)", "TPR on Friday at 1% FPR", cfg.step1_results_path / "fig_step1_tpr1_vs_comm.png")
    save_line_plot(step1_df, "comm_cost_units", "monitor_fpr", "method", "Step 1: Monday Monitor FPR vs Communication Cost", "Communication cost (scalar-equivalent units per sample)", "Measured Monday FPR", cfg.step1_results_path / "fig_step1_monitor_fpr_vs_comm.png")

    run_step2_l2_optimization(cfg)
    step2_df = pd.read_csv(cfg.step2_results_path / "l2_optimization_metrics.csv")
    order = ["L2_mean", "L2_trimmed_mean", "L2_stacking_lr", "L2_attention_ap", "L2_attention_robust"]
    save_grouped_bar_plot(step2_df, "method", "partition_mode", "test_pr_auc", "Step 2: L2 PR-AUC (Random vs AAP)", "L2 fusion method", "PR-AUC on Friday", cfg.step2_results_path / "fig_step2_pr_auc_compare.png", order=order)
    save_grouped_bar_plot(step2_df, "method", "partition_mode", "test_tpr_at_fpr1", "Step 2: L2 TPR@FPR=1% (Random vs AAP)", "L2 fusion method", "TPR on Friday at 1% FPR", cfg.step2_results_path / "fig_step2_tpr1_compare.png", order=order)
    gains_df = pd.read_csv(cfg.step2_results_path / "aap_gain_by_fusion.csv")
    save_gain_bar_plot(gains_df, "method", "tpr_gain_aap_minus_random", "Step 2: TPR Gain from AAP over Random", "L2 fusion method", "TPR gain at 1% FPR", cfg.step2_results_path / "fig_step2_tpr_gain_aap_minus_random.png")

    print("Finished running all experiments.")
