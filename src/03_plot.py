import os
import math
import pandas as pd
import matplotlib.pyplot as plt

IN_PATH = "results/metrics_by_method_calib_monday.csv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# 你可以在这里切换是否在图里显示每个 L0_agent
SHOW_L0_AGENTS = True

def _nice_method_name(m: str) -> str:
    # 可读性更强一点
    m = m.replace("L2_stacking_lr", "L2-stacking")
    m = m.replace("L2_mean", "L2-mean")
    m = m.replace("L1_vote", "L1-vote")
    m = m.replace("Centralized", "Centralized")
    m = m.replace("L3_pca_", "L3-PCA-")
    m = m.replace("L0_", "L0-")
    return m

def _is_fusion_method(method: str) -> bool:
    # fusion methods = Centralized + L1/L2/L3
    if method == "Centralized":
        return True
    if method.startswith("L1_") or method.startswith("L2_") or method.startswith("L3_"):
        return True
    return False

def _annotate_points(ax, df, x_col, y_col, max_annot=30):
    """
    给点加标签。为了不太挤，只标注最重要的点：
    - 一定标注 Centralized / L1 / L2 / L3
    - L0 若允许显示则只标注 PR-AUC 最高的 2 个
    """
    # 优先标注的集合
    priority = []
    for m in df["method"].tolist():
        if m == "Centralized":
            priority.append(m)
        elif m.startswith("L1_") or m.startswith("L2_") or m.startswith("L3_"):
            priority.append(m)

    # 如果包含 L0，只选两条最好的 L0 来标注
    l0 = df[df["method"].str.startswith("L0_")].copy()
    if len(l0) > 0:
        # 用 PR-AUC 排序挑两个
        l0_best = l0.sort_values("test_pr_auc", ascending=False).head(2)["method"].tolist()
        priority += l0_best

    seen = set()
    selected = []
    for m in priority:
        if m in seen:
            continue
        seen.add(m)
        selected.append(m)
        if len(selected) >= max_annot:
            break

    # 按 x 排序，给一点点交错偏移
    sel_df = df[df["method"].isin(selected)].sort_values(x_col)
    for i, r in enumerate(sel_df.itertuples(index=False)):
        x = getattr(r, x_col)
        y = getattr(r, y_col)
        if pd.isna(y):
            continue
        dx = 0.0
        dy = (0.008 if i % 2 == 0 else -0.010)
        label = _nice_method_name(getattr(r, "method"))
        ax.text(x + dx, y + dy, label, fontsize=9)

def _save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    print("Saved:", path)

def main():
    df = pd.read_csv(IN_PATH)

    # 基础清理：按通信成本排序
    df = df.sort_values(["comm_cost_floats", "method"]).reset_index(drop=True)

    # 选择要画的行
    if SHOW_L0_AGENTS:
        plot_df = df.copy()
    else:
        plot_df = df[~df["method"].str.startswith("L0_")].copy()

    # 为了图更清晰，我们额外做一个“只看融合方法”的表
    fusion_df = df[df["method"].apply(_is_fusion_method)].copy()
    fusion_df = fusion_df.sort_values(["comm_cost_floats", "method"])
    fusion_df.to_csv(os.path.join(OUT_DIR, "fusion_only_table.csv"), index=False)

    # ========== 图 1：PR-AUC vs 通信成本 ==========
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    ax1.plot(
        plot_df["comm_cost_floats"],
        plot_df["test_pr_auc"],
        marker="o",
        linewidth=1.6
    )

    ax1.set_xlabel("Communication cost (floats per sample)")
    ax1.set_ylabel("PR-AUC on Friday (higher is better)")
    ax1.set_title("Performance vs Communication: PR-AUC (Test=Friday)")
    ax1.grid(True, alpha=0.3)

    # 标注重点点
    _annotate_points(ax1, plot_df, "comm_cost_floats", "test_pr_auc")

    _save(fig1, "fig_pr_auc_vs_comm.png")

    # ========== 图 2：TPR@FPR=1% vs 通信成本 ==========
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    ax2.plot(
        plot_df["comm_cost_floats"],
        plot_df["test_tpr_at_fpr1"],
        marker="o",
        linewidth=1.6
    )

    ax2.set_xlabel("Communication cost (floats per sample)")
    ax2.set_ylabel("TPR on Friday at FPR=1% (higher is better)")
    ax2.set_title("Operational Performance vs Communication: TPR@FPR=1% (Test=Friday)")
    ax2.grid(True, alpha=0.3)

    _annotate_points(ax2, plot_df, "comm_cost_floats", "test_tpr_at_fpr1")

    _save(fig2, "fig_tpr_at_fpr1_vs_comm.png")

    # ========== 图 3：Monday FPR vs 通信成本 ==========
    # 这张图是“稳定性/误报代价”的关键证据
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.plot(
        plot_df["comm_cost_floats"],
        plot_df["mon_fpr"],
        marker="o",
        linewidth=1.6
    )

    ax3.set_xlabel("Communication cost (floats per sample)")
    ax3.set_ylabel("FPR on Monday (benign-only, lower is better)")
    ax3.set_title("Stability vs Communication: Monday False Positive Rate")
    ax3.grid(True, alpha=0.3)

    _annotate_points(ax3, plot_df, "comm_cost_floats", "mon_fpr")

    _save(fig3, "fig_monday_fpr_vs_comm.png")

    # ========== 额外：输出一个“论文写作友好”的摘要表 ==========
    # 把核心方法挑出来，直接可贴论文/汇报
    keep_methods = []
    for m in df["method"].tolist():
        if m == "Centralized":
            keep_methods.append(m)
        if m in ["L1_vote", "L2_mean", "L2_stacking_lr", "L3_pca_d1", "L3_pca_d4", "L3_pca_d8", "L3_pca_d16", "L3_pca_d32"]:
            keep_methods.append(m)
    if SHOW_L0_AGENTS:
        # 加两条最强 L0
        l0_best = df[df["method"].str.startswith("L0_")].sort_values("test_pr_auc", ascending=False).head(2)["method"].tolist()
        keep_methods += l0_best

    keep_methods = list(dict.fromkeys(keep_methods))  # 去重保持顺序
    summary = df[df["method"].isin(keep_methods)].copy()
    summary = summary.sort_values(["comm_cost_floats", "method"])
    summary.to_csv(os.path.join(OUT_DIR, "paper_summary_table.csv"), index=False)

    print("Saved tables:")
    print(" - results/fusion_only_table.csv")
    print(" - results/paper_summary_table.csv")
    print("\nDone.")

if __name__ == "__main__":
    main()
