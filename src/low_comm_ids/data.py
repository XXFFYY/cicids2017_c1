from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ExperimentConfig

DROP_COLS_CANDIDATES = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Src IP",
    "Dst IP",
}


def load_one_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def infer_day_from_filename(fname: str | Path) -> str:
    base = os.path.basename(str(fname)).lower()
    return base.split("-")[0]


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    label_col = None
    for c in df.columns:
        if c.lower() in {"label", " labels", "labels"}:
            label_col = c
            break
    if label_col is None:
        label_col = "Label"

    df[label_col] = df[label_col].astype(str).str.strip()
    df["y"] = (df[label_col].str.upper() != "BENIGN").astype(int)

    drop_cols = [c for c in df.columns if c in DROP_COLS_CANDIDATES]
    if label_col in df.columns:
        drop_cols.append(label_col)
    df = df.drop(columns=list(set(drop_cols)), errors="ignore")

    y = df["y"].values
    df = df.drop(columns=["y"])
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["y"] = y
    return df


def build_processed_dataset(cfg: ExperimentConfig) -> Path:
    cfg.ensure_dirs()
    paths = sorted(glob.glob(str(cfg.raw_path / "*.csv")))
    if not paths:
        raise RuntimeError(f"No CSV found under {cfg.raw_path}")

    parts = []
    for path in paths:
        day = infer_day_from_filename(path)
        df = load_one_csv(path)
        df = clean_df(df)
        df["day"] = day
        parts.append(df)

    all_df = pd.concat(parts, ignore_index=True)
    all_df.to_parquet(cfg.processed_path, index=False)
    return cfg.processed_path


def load_processed_dataset(cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.processed_path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {cfg.processed_path}. Run scripts/00_build_dataset.py first."
        )
    return pd.read_parquet(cfg.processed_path)


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("y", "day")]


def _maybe_stratify(series: pd.Series):
    return series if series.nunique() > 1 else None


def make_standard_splits(
    df: pd.DataFrame, cfg: ExperimentConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["day"].isin(cfg.train_days)].copy()
    test_df = df[df["day"] == cfg.test_day].copy()
    monday_df = df[df["day"] == cfg.calib_day].copy()

    tr_df, va_df = train_test_split(
        train_df,
        test_size=cfg.val_size,
        stratify=_maybe_stratify(train_df["y"]),
        random_state=cfg.random_state,
    )

    calib_threshold_df, calib_report_df = train_test_split(
        monday_df,
        test_size=cfg.calib_split_size,
        stratify=_maybe_stratify(monday_df["y"]),
        random_state=cfg.random_state,
    )

    return tr_df, va_df, test_df, calib_threshold_df, calib_report_df
