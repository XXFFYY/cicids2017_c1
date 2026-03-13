from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from low_comm_ids import ExperimentConfig
from low_comm_ids.experiment_step1 import run_step1_baselines


if __name__ == "__main__":
    cfg = ExperimentConfig(project_root=ROOT)
    df = run_step1_baselines(cfg)
    print("=" * 80)
    print("Step 1 baseline metrics saved to:", cfg.step1_results_path / "baseline_metrics.csv")
    print("=" * 80)
    print(df.to_string(index=False))
