import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from scipy.special import softmax

# 路径与配置
DATA_PATH = "data_processed/ton_iot_all.parquet"
MODEL_DIR = "models"
OUTPUT_DIR = "results"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "aap_vs_random_poison_test.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_tpr_at_fpr1(y_te, p_te, p_calib):
    # 使用 99% 分位数定位 1% 误报率的阈值
    threshold = np.quantile(p_calib, 0.99)
    y_pred = (p_te >= threshold).astype(int)
    tp = ((y_te == 1) & (y_pred == 1)).sum()
    fn = ((y_te == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def run_experiment(mode, poison_category=None):
    df = pd.read_parquet(DATA_PATH)
    test_df = df[df["day"] == "test"].copy()
    calib_df = df[df["day"] == "calib"].copy()
    train_df = df[df["day"] == "train"].copy()
    
    # 划分验证集用于计算 Attention 权重
    _, va_df = train_test_split(train_df, test_size=0.2, stratify=train_df["y"], random_state=42)
    
    # 加载对应模式的模型
    model_file = os.path.join(MODEL_DIR, f"local_agents_{mode}.joblib")
    if not os.path.exists(model_file):
        print(f"❌ 错误：找不到模型文件 {model_file}，请先运行训练脚本。")
        return None, None

    agents = joblib.load(model_file)
    
    # 模拟故障：毒化特定类别的特征
    if poison_category:
        poison_cols = [c for c in test_df.columns if poison_category in c.lower()]
        for c in poison_cols:
            # 注入 0~1 的随机噪音
            test_df[c] = np.random.rand(len(test_df))
            calib_df[c] = np.random.rand(len(calib_df))
            va_df[c] = np.random.rand(len(va_df))
        print(f"   [!] {mode.upper()} 模式：已毒化 {len(poison_cols)} 个 {poison_category.upper()} 相关特征")

    # 获取所有 Agent 的预测概率
    va_scores, te_scores, ca_scores = [], [], []
    for name in agents:
        cols = agents[name]["cols"]
        m = agents[name]["model"]
        va_scores.append(m.predict_proba(va_df[cols])[:, 1])
        te_scores.append(m.predict_proba(test_df[cols])[:, 1])
        ca_scores.append(m.predict_proba(calib_df[cols])[:, 1])
    
    va_mat = np.column_stack(va_scores)
    te_mat = np.column_stack(te_scores)
    ca_mat = np.column_stack(ca_scores)
    
    # 基于验证集 PR-AUC 的 Attention 融合
    aps = np.array([average_precision_score(va_df["y"], va_mat[:, i]) for i in range(va_mat.shape[1])])
    w = softmax(aps / 0.1) # 温度系数 0.1 强化权重分配
    
    p_te_att = te_mat.dot(w)
    p_ca_att = ca_mat.dot(w)
    
    roc = roc_auc_score(test_df["y"], p_te_att)
    tpr = get_tpr_at_fpr1(test_df["y"], p_te_att, p_ca_att)
    
    return roc, tpr

def main():
    print("🚀 启动对比实验：模拟 HTTP 协议传感器故障下的 AAP vs RANDOM 表现...")
    
    results_list = []
    poison_target = "http"

    for mode in ["aap", "random"]:
        roc, tpr = run_experiment(mode, poison_category=poison_target)
        if roc is not None:
            results_list.append({
                "Partition_Mode": mode.upper(),
                "Poison_Category": poison_target.upper(),
                "Test_ROC_AUC": roc,
                "TPR_at_1percent_FPR": tpr
            })
            print(f"✅ {mode.upper()} 运行完成 | ROC: {roc:.4f} | TPR: {tpr:.4f}")

    # 保存结果到文件
    res_df = pd.DataFrame(results_list)
    res_df.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "="*60)
    print(f"📊 实验数据已成功保存至: {OUTPUT_PATH}")
    print("="*60)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()