import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# 完美对齐你的目录结构
DATA_PATH = "data_processed/cicids2017_all.parquet"
MODEL_DIR = "models"
OUT_JSON = os.path.join(MODEL_DIR, "agent_config_aap.json")

def get_train_data(df):
    """只提取训练集数据（Tue, Wed, Thu），和你的 01 脚本保持绝对一致"""
    train_days = {"tuesday", "wednesday", "thursday"}
    return df[df["day"].isin(train_days)].copy()

def main():
    print("🚀 开始执行 AAP (Auto-Agent Partitioning) 特征自动划分...")
    
    # 1. 加载数据
    print(f"📦 正在加载数据: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    train_df = get_train_data(df)
    
    # 2. 提取特征列（排除标签 y 和日期 day）
    feature_cols = [c for c in train_df.columns if c not in ("y", "day")]
    X_train = train_df[feature_cols]
    
    # 3. 剔除“毫无波澜”的无效特征（标准差为0的常数列）
    # 如果一个特征全是一个值，它无法计算相关性
    std_vals = X_train.std()
    valid_features = std_vals[std_vals > 0].index.tolist()
    X_train = X_train[valid_features]
    print(f"📊 参与聚类的有效特征总数: {len(valid_features)} (原总数: {len(feature_cols)})")

    # 4. 计算特征之间的相关系数矩阵
    # 取绝对值是因为：无论正相关还是负相关，只要有强关联，就说明它们掌握着重叠的信息
    print("🧮 正在计算特征相关性矩阵 (Pearson Correlation)...")
    corr_matrix = X_train.corr().abs().fillna(0)
    
    # 5. 将相关性转化为“距离”
    # 相关性越强（接近1），距离越近（接近0）；相关性越弱（接近0），距离越远（接近1）
    dist_matrix = 1.0 - corr_matrix
    
    # 6. 使用层次聚类 (Agglomerative Clustering) 将特征分为 4 个 Agent
    # linkage='complete' 倾向于把彼此之间都高度相关的特征抱团在一起
    n_agents = 4
    print(f"🤖 正在执行层次聚类，目标 Agent 数量: {n_agents}...")
    cluster = AgglomerativeClustering(
        n_clusters=n_agents, 
        metric='precomputed', 
        linkage='complete'
    )
    labels = cluster.fit_predict(dist_matrix)

    # 7. 构建新的 Agent 字典
    aap_groups = {}
    for i in range(n_agents):
        agent_name = f"AAP_Agent_{i}"
        # 找出被分到第 i 类的所有特征
        cluster_features = [valid_features[j] for j, label in enumerate(labels) if label == i]
        aap_groups[agent_name] = cluster_features
        print(f"✅ {agent_name} 分配到了 {len(cluster_features)} 个特征。")

    # 8. 保存为 JSON 文件，供后续 01 脚本读取
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(aap_groups, f, indent=4)
    
    print(f"\n🎉 大功告成！AAP 配置文件已保存至: {OUT_JSON}")

if __name__ == "__main__":
    main()