import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.special import softmax

DATA_PATH = "data_processed/ton_iot_all.parquet"
MODEL_DIR = "models"
OUTPUT_PATH = "results/aap_vs_random_comparison.csv"
os.makedirs("results", exist_ok=True)

def get_tpr_at_fpr_robust(y_te, p_te, p_calib, target_fpr=0.01):
    threshold = np.quantile(p_calib, 1.0 - target_fpr)
    y_pred = (p_te >= threshold).astype(int)
    tp = ((y_te == 1) & (y_pred == 1)).sum()
    fn = ((y_te == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def main():
    df = pd.read_parquet(DATA_PATH)
    train_df = df[df["day"] == "train"].copy()
    test_df = df[df["day"] == "test"].copy()
    calib_df = df[df["day"] == "calib"].copy()
    y_te, y_calib = test_df["y"].values, calib_df["y"].values
    
    _, va_df = train_test_split(train_df, test_size=0.2, stratify=train_df["y"], random_state=42)
    y_va = va_df["y"].values

    final_results = []

    for mode in ["aap", "random"]:
        print(f"🔍 正在评估 [{mode.upper()}] 模式...")
        agents = joblib.load(os.path.join(MODEL_DIR, f"local_agents_{mode}.joblib"))
        agent_names = list(agents.keys())
        
        va_scores, te_scores, ca_scores = [], [], []
        for name in agent_names:
            cols = agents[name]["cols"]
            m = agents[name]["model"]
            va_scores.append(m.predict_proba(va_df[cols])[:, 1])
            te_scores.append(m.predict_proba(test_df[cols])[:, 1])
            ca_scores.append(m.predict_proba(calib_df[cols])[:, 1])
        
        va_mat = np.column_stack(va_scores)
        te_mat = np.column_stack(te_scores)
        ca_mat = np.column_stack(ca_scores)

        # 核心：Attention 融合
        aps = np.array([average_precision_score(y_va, va_mat[:, i]) for i in range(len(agent_names))])
        w = softmax(aps / 0.1)
        p_te_att = te_mat.dot(w)
        p_ca_att = ca_mat.dot(w)

        # 指标计算
        roc = roc_auc_score(y_te, p_te_att)
        pr = average_precision_score(y_te, p_te_att)
        tpr1 = get_tpr_at_fpr_robust(y_te, p_te_att, p_ca_att)

        final_results.append({
            "Partition_Method": mode.upper(),
            "Fusion_Method": "Attention",
            "Test_ROC_AUC": roc,
            "Test_PR_AUC": pr,
            "TPR_at_1percent_FPR": tpr1
        })

    # 输出对比表格
    res_df = pd.DataFrame(final_results)
    res_df.to_csv(OUTPUT_PATH, index=False)
    print("\n" + "="*60)
    print("AAP 消融实验大结局：AAP vs 随机分区")
    print("="*60)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()