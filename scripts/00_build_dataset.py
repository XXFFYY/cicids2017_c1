from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from low_comm_ids import ExperimentConfig
from low_comm_ids.data import build_processed_dataset


if __name__ == "__main__":
    cfg = ExperimentConfig(project_root=ROOT)
    out_path = build_processed_dataset(cfg)
    print(f"Saved processed dataset to: {out_path}")
