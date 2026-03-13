from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def nice_method_name(name: str) -> str:
    mapping = {
        "Best_L0": "Best-L0",
        "L1_vote": "L1-vote",
        "L2_mean": "L2-mean",
        "L2_trimmed_mean": "L2-trimmed",
        "L2_stacking_lr": "L2-stacking",
        "L2_attention_ap": "L2-attention",
        "L2_attention_robust": "L2-robust-attn",
        "Centralized": "Centralized",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("L3_pca_d"):
        return name.replace("L3_pca_d", "L3-PCA-")
    if name.startswith("L0_"):
        return name.replace("L0_", "L0-")
    return name


def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    label_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
    annotate: bool = True,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(df[x_col], df[y_col], marker="o", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if annotate:
        for i, row in enumerate(df.itertuples(index=False)):
            dx = 0.0
            dy = 0.008 if i % 2 == 0 else -0.012
            ax.text(
                getattr(row, x_col) + dx,
                getattr(row, y_col) + dy,
                nice_method_name(getattr(row, label_col)),
                fontsize=9,
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_grouped_bar_plot(
    df: pd.DataFrame,
    index_col: str,
    group_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
    order: Sequence[str] | None = None,
) -> None:
    pivot = df.pivot(index=index_col, columns=group_col, values=value_col)
    if order is not None:
        pivot = pivot.reindex(order)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111)
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title=group_col)
    ax.set_xticklabels([nice_method_name(str(x.get_text())) for x in ax.get_xticklabels()], rotation=30, ha="right")

    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            f"{height:.3f}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 4),
            textcoords="offset points",
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_gain_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str | Path,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels([nice_method_name(str(x.get_text())) for x in ax.get_xticklabels()], rotation=30, ha="right")
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.3f}",
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 4),
            textcoords="offset points",
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
