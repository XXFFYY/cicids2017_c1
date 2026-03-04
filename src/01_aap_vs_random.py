import os
import json
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

DATA_PATH = "data_processed/ton_iot_all.parquet"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_one_agent(X_train, y_train):
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=64, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    return model

def main():
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("y", "day")]
    train_df = df[df["day"] == "train"].copy()
    tr_df, va_df = train_test_split(train_df, test_size=0.2, stratify=train_df["y"], random_state=42)

    # 提取全局共享特征 (Top 3 重要性)
    lgbm_global = LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
    lgbm_global.fit(tr_df[feature_cols], tr_df["y"].values)
    top_indices = np.argsort(lgbm_global.feature_importances_)[::-1][:3]
    top_shared_features = [feature_cols[i] for i in top_indices]
    remaining_features = [c for c in feature_cols if c not in top_shared_features]

    # === 核心：两种模式的特征分组 ===
    partition_modes = ["aap", "random"]
    
    for mode in partition_modes:
        print(f"\n🚀 正在以 [{mode.upper()}] 模式进行特征分区训练...")
        groups = {f"Agent_{i}": [] for i in range(4)}
        
        if mode == "aap":
            # AAP 逻辑：相关性聚类
            corr_matrix = tr_df[remaining_features].corr().fillna(0).values
            pca = PCA(n_components=min(10, len(remaining_features)), random_state=42)
            Z = pca.fit_transform(corr_matrix)
            clusters = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(Z)
            for f_name, c_id in zip(remaining_features, clusters):
                groups[f"Agent_{c_id}"].append(f_name)
        else:
            # Random 模式：完全随机分配
            rem_feat_shuffled = remaining_features.copy()
            np.random.seed(42); np.random.shuffle(rem_feat_shuffled)
            n = len(rem_feat_shuffled) // 4
            for i in range(4):
                groups[f"Agent_{i}"] = rem_feat_shuffled[i*n : (i+1)*n]

        # 训练并保存
        current_models = {}
        for agent_name, cols in groups.items():
            full_cols = top_shared_features + cols
            print(f"  正在训练 {agent_name} (特征数: {len(full_cols)})...")
            m = train_one_agent(tr_df[full_cols], tr_df["y"].values)
            current_models[agent_name] = {"model": m, "cols": full_cols}
        
        joblib.dump(current_models, os.path.join(MODEL_DIR, f"local_agents_{mode}.joblib"))

    print(f"\n✅ 双模式训练完成！模型已保存至 {MODEL_DIR}")

if __name__ == "__main__":
    main()