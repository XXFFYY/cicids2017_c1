from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from low_comm_ids import ExperimentConfig
from low_comm_ids.experiment_step2 import run_step2_l2_optimization


if __name__ == "__main__":
    cfg = ExperimentConfig(project_root=ROOT)
    df = run_step2_l2_optimization(cfg)
    print("=" * 80)
    print("Step 2 L2 optimization metrics saved to:", cfg.step2_results_path / "l2_optimization_metrics.csv")
    print("=" * 80)
    print(df.to_string(index=False))
