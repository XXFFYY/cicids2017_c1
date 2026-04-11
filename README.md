# 低通信多智能体协作入侵检测：重构版项目

这个重构版严格围绕两步实验主线来组织：

## 实验主线

### Step 1：基线实验
目标：证明在 `L0 / L1 / L2 / L3 / Centralized` 这些基线中，`L2` 在低通信成本下有较好的整体准确率表现，但在 `TPR@FPR=1%` 这个更严格的运行指标上仍然不够理想。

输出：
- `results/step1/baseline_metrics.csv`
- `results/step1/baseline_plot_table.csv`
- `results/step1/fig_step1_pr_auc_vs_comm.png`
- `results/step1/fig_step1_tpr1_vs_comm.png`
- `results/step1/fig_step1_monitor_fpr_vs_comm.png`

### Step 2：围绕 L2 做优化
目标：在保留 `L2` 低通信优势的前提下，引入 `AAP + Attention`，重点提升 `TPR@FPR=1%`。

输出：
- `results/step2/l2_optimization_metrics.csv`
- `results/step2/l2_attention_weights.csv`
- `results/step2/aap_gain_by_fusion.csv`
- `results/step2/l2_gain_breakdown.csv`
- `results/step2/fig_step2_pr_auc_compare.png`
- `results/step2/fig_step2_tpr1_compare.png`
- `results/step2/fig_step2_tpr_gain_aap_minus_random.png`
- `results/step2/fig_step2_tpr_gain_within_aap.png`

---

## 为什么这次重构会更清晰

原始项目里，训练、融合、AAP、画图、鲁棒性测试是交叉写在多个脚本里的，容易出现两个问题：

1. **脚本职责混在一起**：一个脚本既做训练又做实验逻辑又做保存。
2. **实验叙事不够清楚**：你真正想讲的是“先做基线，再对 L2 做优化”，但原始脚本文件名不直接体现这个故事线。

这次重构后，结构按“实验阶段”来组织，而不是按“零散功能”来组织。

---

## 新项目结构

```text
low_comm_ids_refactor/
├─ README.md
├─ requirements.txt
├─ scripts/
│  ├─ 00_build_dataset.py
│  ├─ 10_run_step1_baselines.py
│  ├─ 11_plot_step1.py
│  ├─ 20_run_step2_l2_optimization.py
│  ├─ 21_plot_step2.py
│  └─ 99_run_all.py
├─ src/
│  └─ low_comm_ids/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ data.py
│     ├─ partitioning.py
│     ├─ models.py
│     ├─ fusion.py
│     ├─ metrics.py
│     ├─ plotting.py
│     ├─ experiment_step1.py
│     └─ experiment_step2.py
├─ data_processed/
├─ models/
│  ├─ step1/
│  └─ step2/
└─ results/
   ├─ step1/
   └─ step2/
```

---

## 每个模块的作用

### `data.py`
负责：
- 原始 CSV 清洗
- 标签二值化
- 删除明显泄漏身份的信息列
- 保存 parquet
- 按 `Tue/Wed/Thu -> train`、`Friday -> test`、`Monday -> calibration` 划分
- 把 Monday 再拆成两半：
  - 一半专门定阈值
  - 一半专门报告真实 FPR

这样比“用同一份 Monday 既定阈值又测 FPR”更严谨。

### `partitioning.py`
负责：
- 共享特征选择
- `Random` 分组
- `AAP` 分组

这里把 **AAP 固定为你说的 04 号脚本那条思路**：
- 先算特征相关性
- 再转成距离
- 再做层次聚类

### `models.py`
负责：
- 训练本地 Agent
- 训练集中式模型
- 训练 `L3` 的 PCA 压缩通信模型

### `fusion.py`
负责：
- `L1_vote`
- `L2_mean`
- `L2_trimmed_mean`
- `L2_stacking_lr`
- `L2_attention_ap`
- `L2_attention_robust`

### `metrics.py`
负责：
- PR-AUC
- ROC-AUC
- `TPR@FPR=1%`
- Monday 监控 FPR
- 最佳 `L0` 摘要行

### `plotting.py`
负责统一出图，避免每个实验脚本自己画一套。

---

## 两步实验具体定义

## Step 1：基线比较

### 包含的方法
- `L0_Agent_i`
- `L1_vote`
- `L2_mean`
- `L2_trimmed_mean`
- `L2_stacking_lr`
- `L3_pca_d1`
- `L3_pca_d4`
- `L3_pca_d8`
- `L3_pca_d16`
- `L3_pca_d32`
- `Centralized`

### 解释
- `L0`：单个 agent 独立判断
- `L1`：只做硬投票
- `L2`：只上传每个 agent 的分数，再做分数级融合
- `L3`：上传压缩后的中间表示（PCA 维度越大通信越高）
- `Centralized`：直接看全部特征

### 你要讲的论文故事
- 看 `fig_step1_pr_auc_vs_comm.png`：说明 `L2` 在低通信成本区域很有竞争力
- 看 `fig_step1_tpr1_vs_comm.png`：说明 `L2` 在严格误报约束下还有明显提升空间

---

## Step 2：L2 优化实验

### 包含的优化因素
- 分区方式：`Random` vs `AAP`
- 融合方式：
  - `L2_mean`
  - `L2_trimmed_mean`
  - `L2_stacking_lr`
  - `L2_attention_ap`
  - `L2_attention_robust`

### 你要讲的论文故事
- `fig_step2_pr_auc_compare.png`：看整体准确率是否保持或提升
- `fig_step2_tpr1_compare.png`：看 `AAP + Attention` 是否显著提高 `TPR@FPR=1%`
- `fig_step2_tpr_gain_aap_minus_random.png`：看 AAP 相比 Random 的增益
- `fig_step2_tpr_gain_within_aap.png`：看在 AAP 条件下，哪种 L2 融合最有效

---

## 运行顺序

### 方式 1：一步一步跑

```bash
cd low_comm_ids_refactor
python scripts/00_build_dataset.py
python scripts/10_run_step1_baselines.py
python scripts/11_plot_step1.py
python scripts/20_run_step2_l2_optimization.py
python scripts/21_plot_step2.py
```

### 方式 2：全部自动跑

```bash
cd low_comm_ids_refactor
python scripts/99_run_all.py
```

---

