import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 你的两个结果文件路径
FILE_MANUAL = "results/metrics_manual.csv"
FILE_AAP = "results/metrics_aap.csv"
OUT_DIR = "results"

def load_and_prep(file_path, prefix):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}，请检查路径和文件名！")
    df = pd.read_csv(file_path)
    
    # 提取我们关心的对比指标
    methods = ["L1_vote", "L2_mean", "L2_stacking_lr", "L3_pca_d8", "L3_pca_d16", "L3_pca_d32", "Centralized_restricted"]
    df_filtered = df[df["method"].isin(methods)].copy()
    
    # 找出 L0 单兵作战的最高分（代表该分组下的最强特种兵）
    l0_best = df[df["method"].str.startswith("L0_")].sort_values("test_pr_auc", ascending=False).iloc[0]
    l0_best_row = pd.DataFrame([l0_best])
    l0_best_row["method"] = "Best_L0_Single"
    
    df_filtered = pd.concat([l0_best_row, df_filtered], ignore_index=True)
    df_filtered["Group"] = prefix
    return df_filtered

def plot_comparison_bar(df_all, metric_col, title, ylabel, save_name):
    # 整理数据格式以便于画图
    pivot_df = df_all.pivot(index="method", columns="Group", values=metric_col)
    
    # 强行排个序：按通信成本从小到大（理论上）
    order = ["Best_L0_Single", "L1_vote", "L2_mean", "L2_stacking_lr", 
             "L3_pca_d8", "L3_pca_d16", "L3_pca_d32", "Centralized_restricted"]
    pivot_df = pivot_df.reindex(order)

    # 画并排柱状图
    ax = pivot_df.plot(kind='bar', figsize=(12, 6), color=['#1f77b4', '#ff7f0e'], alpha=0.85, edgecolor='black')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Communication / Fusion Method", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Grouping Strategy", fontsize=11)
    
    # 在柱子顶端标上具体数字
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=9, xytext=(0, 5), 
                    textcoords='offset points')
        
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, save_name)
    plt.savefig(out_path, dpi=200)
    print(f"✅ 成功生成对比图: {out_path}")
    plt.close()

def main():
    print("📊 正在加载数据进行 Manual vs AAP 的巅峰对决...")
    df_manual = load_and_prep(FILE_MANUAL, "Manual (Part 1)")
    df_aap = load_and_prep(FILE_AAP, "AAP (Part 2)")
    
    df_all = pd.concat([df_manual, df_aap], ignore_index=True)

    # 画图 1：PR-AUC 对比
    plot_comparison_bar(df_all, "test_pr_auc", 
                        "Comparison of PR-AUC: Manual vs AAP", 
                        "PR-AUC (Higher is Better)", 
                        "fig_compare_prauc.png")
    
    # 画图 2：TPR@FPR=1% 对比
    plot_comparison_bar(df_all, "test_tpr_at_fpr1", 
                        "Comparison of TPR@FPR=1%: Manual vs AAP", 
                        "TPR (Higher is Better)", 
                        "fig_compare_tpr.png")
                        
    print("\n🎉 对比完成！请去 results 文件夹查看你的两张新图！")

if __name__ == "__main__":
    main()