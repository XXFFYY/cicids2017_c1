from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class ExperimentConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    raw_dir: str = "data_raw/CICIDS2017/MachineLearningCSV"
    processed_relpath: str = "data_processed/cicids2017_all.parquet"
    models_dir: str = "models"
    results_dir: str = "results"

    train_days: Tuple[str, ...] = ("tuesday", "wednesday", "thursday")
    test_day: str = "friday"
    calib_day: str = "monday"

    n_agents: int = 4
    n_shared_features: int = 3
    val_size: float = 0.2
    calib_split_size: float = 0.5
    target_fpr: float = 0.01
    random_state: int = 42

    l3_dims: Tuple[int, ...] = (1, 4, 8, 16, 32)
    attention_temperature: float = 0.1
    robust_health_floor: float = 0.05
    l1_threshold: float = 0.5

    lgbm_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    )
    global_lgbm_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 50,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    )

    @property
    def raw_path(self) -> Path:
        return self.project_root / self.raw_dir

    @property
    def processed_path(self) -> Path:
        return self.project_root / self.processed_relpath

    @property
    def models_path(self) -> Path:
        return self.project_root / self.models_dir

    @property
    def results_path(self) -> Path:
        return self.project_root / self.results_dir

    @property
    def step1_results_path(self) -> Path:
        return self.results_path / "step1"

    @property
    def step2_results_path(self) -> Path:
        return self.results_path / "step2"

    @property
    def step1_models_path(self) -> Path:
        return self.models_path / "step1"

    @property
    def step2_models_path(self) -> Path:
        return self.models_path / "step2"

    def ensure_dirs(self) -> None:
        for path in [
            self.processed_path.parent,
            self.models_path,
            self.results_path,
            self.step1_results_path,
            self.step2_results_path,
            self.step1_models_path,
            self.step2_models_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["project_root"] = str(self.project_root)
        data["raw_path"] = str(self.raw_path)
        data["processed_path"] = str(self.processed_path)
        data["models_path"] = str(self.models_path)
        data["results_path"] = str(self.results_path)
        return data
