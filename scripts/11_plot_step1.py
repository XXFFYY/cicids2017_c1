from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from low_comm_ids import ExperimentConfig
from low_comm_ids.plotting import save_line_plot


if __name__ == "__main__":
    cfg = ExperimentConfig(project_root=ROOT)
    in_path = cfg.step1_results_path / "baseline_plot_table.csv"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run scripts/10_run_step1_baselines.py first.")

    df = pd.read_csv(in_path).sort_values(["comm_cost_units", "method"]).reset_index(drop=True)

    save_line_plot(
        df=df,
        x_col="comm_cost_units",
        y_col="test_pr_auc",
        label_col="method",
        title="Step 1: PR-AUC vs Communication Cost",
        xlabel="Communication cost (scalar-equivalent units per sample)",
        ylabel="PR-AUC on Friday",
        out_path=cfg.step1_results_path / "fig_step1_pr_auc_vs_comm.png",
    )
    save_line_plot(
        df=df,
        x_col="comm_cost_units",
        y_col="test_tpr_at_fpr1",
        label_col="method",
        title="Step 1: TPR@FPR=1% vs Communication Cost",
        xlabel="Communication cost (scalar-equivalent units per sample)",
        ylabel="TPR on Friday at 1% FPR",
        out_path=cfg.step1_results_path / "fig_step1_tpr1_vs_comm.png",
    )
    save_line_plot(
        df=df,
        x_col="comm_cost_units",
        y_col="monitor_fpr",
        label_col="method",
        title="Step 1: Monday Monitor FPR vs Communication Cost",
        xlabel="Communication cost (scalar-equivalent units per sample)",
        ylabel="Measured Monday FPR",
        out_path=cfg.step1_results_path / "fig_step1_monitor_fpr_vs_comm.png",
    )

    print("Saved step 1 figures to:", cfg.step1_results_path)
