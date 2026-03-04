import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.special import softmax

# 1. 路径与配置
DATA_PATH = "data_processed/ton_iot_all.parquet"
MODEL_DIR = "models"
OUTPUT_PATH = "results/drift_robustness_results.csv"
os.makedirs("results", exist_ok=True)

# 2. 加载数据
print("正在加载数据...")
df = pd.read_parquet(DATA_PATH)

# 严格遵循无菌划分逻辑
def get_splits(df):
    train_df = df[df["day"] == "train"].copy()
    test_df = df[df["day"] == "test"].copy()
    calib_df = df[df["day"] == "calib"].copy() # 100% 正常流量
    return train_df, test_df, calib_df

train_df, test_df, calib_df = get_splits(df)
y_te = test_df["y"].values

# 准备两组验证集（用于模拟故障感知）
va1_df = train_df.sample(frac=0.2, random_state=1) 
va2_df = train_df.drop(va1_df.index).sample(frac=0.2, random_state=2)

# 3. 加载本地 Agent
agents = joblib.load(os.path.join(MODEL_DIR, "local_agents.joblib"))
agent_names = list(agents.keys())

def get_preds(df_target, agents, poison_idx=None):
    scores = []
    for i, name in enumerate(agent_names):
        p = agents[name]["model"].predict_proba(df_target[agents[name]["cols"]])[:, 1]
        if poison_idx is not None and i == poison_idx:
            p = np.random.rand(len(p)) # 注入随机噪音
        scores.append(p)
    return np.column_stack(scores)

def get_tpr_at_fpr1(y_te, p_te, p_calib):
    """
    使用百分位数在纯正常流量校准集上定阈值
    """
    threshold = np.quantile(p_calib, 0.99) # 1% 误报门槛
    y_pred = (p_te >= threshold).astype(int)
    tp = ((y_te == 1) & (y_pred == 1)).sum()
    fn = ((y_te == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# === 模拟阶段 1: 环境中 Agent 2 故障 (Stacking 在此时训练) ===
print("🛠️  Phase 1: Agent 2 故障中，Stacking 正在固化权重...")
S_va1 = get_preds(va1_df, agents, poison_idx=2)
lr_fixed = LogisticRegression().fit(S_va1, va1_df["y"].values)

# === 模拟阶段 2: 环境发生漂移 (Agent 2 修复, Agent 1 突然被劫持) ===
print("⚠️  Phase 2: 发生环境漂移！Agent 2 修复，Agent 1 变为噪音源...")
S_te = get_preds(test_df, agents, poison_idx=1) 
S_calib = get_preds(calib_df, agents, poison_idx=1) 
S_va2 = get_preds(va2_df, agents, poison_idx=1) # 实时验证流

results = []

# --- 1. L2_Mean (基准) ---
p_te_m = S_te.mean(axis=1)
p_ca_m = S_calib.mean(axis=1)
results.append({
    "Method": "L2_Mean (Static)",
    "Condition": "Phase 2 Drift",
    "ROC_AUC": roc_auc_score(y_te, p_te_m),
    "TPR_at_1percent_FPR": get_tpr_at_fpr1(y_te, p_te_m, p_ca_m)
})

# --- 2. L2_Stacking (权重僵化) ---
p_te_s = lr_fixed.predict_proba(S_te)[:, 1]
p_ca_s = lr_fixed.predict_proba(S_calib)[:, 1]
results.append({
    "Method": "L2_Stacking (Rigid)",
    "Condition": "Phase 2 Drift",
    "ROC_AUC": roc_auc_score(y_te, p_te_s),
    "TPR_at_1percent_FPR": get_tpr_at_fpr1(y_te, p_te_s, p_ca_s)
})

# --- 3. L2_Attention (自适应自愈) ---
aps2 = np.array([average_precision_score(va2_df["y"], S_va2[:, i]) for i in range(4)])
w_adaptive = softmax(aps2 / 0.1)
p_te_a = S_te.dot(w_adaptive)
p_ca_a = S_calib.dot(w_adaptive)
results.append({
    "Method": "L2_Attention (Adaptive)",
    "Condition": "Phase 2 Drift",
    "ROC_AUC": roc_auc_score(y_te, p_te_a),
    "TPR_at_1percent_FPR": get_tpr_at_fpr1(y_te, p_te_a, p_ca_a)
})

# 4. 保存与输出
res_df = pd.DataFrame(results)
res_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*75)
print(f"✅ 动态漂移实验结果已保存至: {OUTPUT_PATH}")
print("="*75)
print(res_df.to_string(index=False))
print("="*75)