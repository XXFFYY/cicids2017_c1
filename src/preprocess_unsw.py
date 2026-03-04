import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_unsw(train_path, test_path, out_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    df = pd.concat([train, test], ignore_index=True)

    # 1. 统一标签：UNSW 的标签列叫 'label' (0为正常, 1为攻击)
    df = df.rename(columns={'label': 'y'})
    
    # 2. 模拟 'day' 列：为了兼容你的 get_splits 逻辑
    # 我们把原始的训练集标记为 'train_phase'，测试集标记为 'test_phase'
    df['day'] = 'train_phase'
    df.iloc[len(train):, df.columns.get_loc('day')] = 'test_phase'

    # 3. 处理类别型特征 (Label Encoding)
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 4. 剔除无效列 (如 ID 或具体的攻击分类名 'attack_cat')
    # 【核心修改点】：把这两个会导致“过拟合”的强力物理特征删掉
    # sttl: Source to destination time to live
    # ct_state_ttl: No. for each state according to specific range of values for source/destination time to live
    drop_cols = ['id', 'attack_cat', 'sttl', 'ct_state_ttl'] 
    
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 5. 保存为 Parquet
    df.to_parquet(out_path, index=False)
    print(f"✅ UNSW-NB15 预处理完成，保存至: {out_path}")

# 使用示例
# preprocess_unsw('UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv', 'data_processed/unsw_all.parquet')
# 在 src/preprocess_unsw.py 的最后一行加上：
if __name__ == "__main__":
    preprocess_unsw(
        'data_raw/UNSW_NB15/UNSW_NB15_training-set.csv', 
        'data_raw/UNSW_NB15/UNSW_NB15_testing-set.csv', 
        'data_processed/unsw_all.parquet'
    )