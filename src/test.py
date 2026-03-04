import pandas as pd

df = pd.read_parquet("data_processed/cicids2017_all.parquet")
print(df["day"].value_counts())
