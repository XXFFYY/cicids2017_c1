
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(".")
CSV_PATH = BASE_DIR / "results" / "robustness_summary.csv"
OUT_DIR = BASE_DIR / "robustness_appendix_fixed"
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

# ---------------------------
# Figure 1: calibration drift
# ---------------------------
cal = df[df["scenario_family"] == "calibration"].copy()
cal_order = ["monday_swap", "friday_halfswap"]
label_map = {
    "monday_swap": "Monday halves swapped",
    "friday_halfswap": "Friday half-swap",
}
obj_map = {
    ("aap", "Global_Best_L0"): "AAP-L0",
    ("aap", "Global_Best_Pair_L2"): "AAP-Pair",
    ("supervised", "Global_Best_L0"): "Supervised-L0",
    ("supervised", "Global_Best_Pair_L2"): "Supervised-Pair",
}

cal["series"] = cal.apply(lambda r: obj_map[(r["mode"], r["object"])], axis=1)
cal["scenario_label"] = cal["scenario"].map(label_map)

pivot_cal = (
    cal.pivot_table(index="series", columns="scenario_label", values="delta_tpr", aggfunc="mean")
    .reindex(["AAP-L0", "AAP-Pair", "Supervised-L0", "Supervised-Pair"])
    [["Monday halves swapped", "Friday half-swap"]]
)

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(pivot_cal.index))
width = 0.35
cols = list(pivot_cal.columns)
for i, col in enumerate(cols):
    ax.bar(x + (i - 0.5) * width, pivot_cal[col].values, width=width, label=col)

ax.axhline(0, linewidth=1)
ax.set_ylabel("Delta TPR@1%FPR")
ax.set_title("Calibration drift robustness")
ax.set_xticks(x)
ax.set_xticklabels(pivot_cal.index, rotation=0)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "appendix_calibration_drift_delta_tpr.png", dpi=220)
plt.close(fig)

# ---------------------------------
# Figure 2: local perturbation loss
# pair rows are aggregated over hit-agent
# ---------------------------------
pert = df[df["scenario_family"] == "perturb"].copy()

def canonical_scenario(s: str) -> str:
    if s.startswith("feature_dropout_30"):
        return "Feature dropout 30%"
    if s.startswith("feature_dropout_50"):
        return "Feature dropout 50%"
    if s.startswith("score_noise_003"):
        return "Score noise 0.03"
    return s

pert["scenario_label"] = pert["scenario"].apply(canonical_scenario)
pert["series"] = pert.apply(lambda r: obj_map[(r["mode"], r["object"])], axis=1)

# Aggregate pair perturbations over hit-agent. Mean is the default choice.
pivot_pert = (
    pert.groupby(["series", "scenario_label"], as_index=False)["delta_tpr"].mean()
    .pivot(index="series", columns="scenario_label", values="delta_tpr")
    .reindex(["AAP-L0", "AAP-Pair", "Supervised-L0", "Supervised-Pair"])
    [["Feature dropout 30%", "Feature dropout 50%", "Score noise 0.03"]]
)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(pivot_pert.index))
width = 0.24
cols = list(pivot_pert.columns)
for i, col in enumerate(cols):
    ax.bar(x + (i - 1) * width, pivot_pert[col].values, width=width, label=col)

ax.axhline(0, linewidth=1)
ax.set_ylabel("Mean Delta TPR@1%FPR")
ax.set_title("Local perturbation robustness (pair rows averaged over hit-agent)")
ax.set_xticks(x)
ax.set_xticklabels(pivot_pert.index, rotation=0)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_DIR / "appendix_local_perturb_delta_tpr.png", dpi=220)
plt.close(fig)

# Optional: export the aggregated tables for the paper appendix
pivot_cal.reset_index().to_csv(OUT_DIR / "appendix_calibration_drift_table.csv", index=False)
pivot_pert.reset_index().to_csv(OUT_DIR / "appendix_local_perturb_table.csv", index=False)

print("Wrote:", OUT_DIR)
