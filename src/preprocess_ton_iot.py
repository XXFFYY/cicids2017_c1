import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_ton_iot(input_path, out_path):
    print(f"正在读取 {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    
    # 统一标签名
    if 'label' in df.columns:
        df = df.rename(columns={'label': 'y'})
    
    # === 1. 极其严苛的 Domain Split (零日隔离 + 独立校准集) ===
    df['day'] = 'unknown'
    np.random.seed(42)
    
    # A. 正常流量 (normal/0): 60% 训练, 20% 校准(calib), 20% 测试
    normal_mask = (df['y'] == 0)
    df.loc[normal_mask, 'day'] = np.random.choice(
        ['train', 'calib', 'test'], size=normal_mask.sum(), p=[0.6, 0.2, 0.2]
    )
    
    # B. 攻击流量 (1): 按攻击家族物理隔离
    # 挑 3 个最狡猾的作为零日攻击 (Zero-Day) 留给考卷
    zero_day_attacks = ['ransomware', 'backdoor', 'injection'] 
    attack_mask = (df['y'] == 1)
    is_zero_day = df['type'].astype(str).str.lower().isin(zero_day_attacks)
    
    df.loc[attack_mask & is_zero_day, 'day'] = 'test'   # 没见过的去考试
    df.loc[attack_mask & ~is_zero_day, 'day'] = 'train' # 见过的进训练集

    # === 2. 拔掉作弊轮：删除所有上帝视角统计特征 ===
    drop_cols = [
        'src_ip', 'src_port', 'dst_ip', 'dst_port', 'type', # 身份和标签泄露
        'duration', 'src_bytes', 'dst_bytes', 'missed_bytes', 
        'src_pkts', 'dst_pkts', 'src_ip_bytes', 'dst_ip_bytes' # 物理统计泄露
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # === 3. 严谨的防泄露 Encoder (未知=UNK) ===
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'day']
    
    train_calib_mask = df['day'].isin(['train', 'calib'])
    
    print(f"🔄 正在处理 {len(cat_cols)} 个字符串特征 (使用 UNK 机制)...")
    for col in cat_cols:
        df[col] = df[col].astype(str)
        # 只从历史(train+calib)中学习词表
        known_cats = set(df.loc[train_calib_mask, col].unique())
        
        # 将所有不认识的新东西标记为 UNK
        df[col] = df[col].apply(lambda x: x if x in known_cats else 'UNK')
        
        encoder = LabelEncoder()
        encoder.fit(list(known_cats) + ['UNK'])
        df[col] = encoder.transform(df[col])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    
    print(f"✅ 训练集(Train): {len(df[df['day']=='train'])} 条")
    print(f"✅ 校准集(Calib/只含正常): {len(df[df['day']=='calib'])} 条")
    print(f"✅ 测试集(Test/含ZeroDay): {len(df[df['day']=='test'])} 条")
    print("✅ 无菌数据已保存！")

if __name__ == "__main__":
    preprocess_ton_iot('data_raw/TON_IoT/ton_iot.csv', 'data_processed/ton_iot_all.parquet')