import os
import glob
import numpy as np
import pandas as pd

RAW_DIR = "data_raw/CICIDS2017/MachineLearningCSV"
OUT_DIR = "data_processed"
os.makedirs(OUT_DIR, exist_ok=True)

DROP_COLS_CANDIDATES = {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Src IP", "Dst IP"}

def load_one_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def infer_day_from_filename(fname: str) -> str:
    base = os.path.basename(fname).lower()
    return base.split("-")[0]  # monday/tuesday/...



def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # unify label col name
    label_col = None
    for c in df.columns:
        if c.lower() in {"label", " labels", "labels"}:
            label_col = c
            break
    if label_col is None:
        # most CICIDS2017 CSV uses "Label"
        label_col = "Label"
    df[label_col] = df[label_col].astype(str).str.strip()

    # binary label
    df["y"] = (df[label_col].str.upper() != "BENIGN").astype(int)

    # drop obvious identifiers if present
    drop_cols = [c for c in df.columns if c in DROP_COLS_CANDIDATES]
    if label_col in df.columns:
        drop_cols.append(label_col)
    df = df.drop(columns=list(set(drop_cols)), errors="ignore")

    # keep numeric only (except y)
    y = df["y"].values
    df = df.drop(columns=["y"])
    # convert to numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # handle inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    
    df.columns = [c.replace(" ", "_") for c in df.columns]

    df["y"] = y
    return df

def main():
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not paths:
        raise RuntimeError(f"No CSV found under {RAW_DIR}")

    parts = []
    for p in paths:
        day = infer_day_from_filename(p)
        df = load_one_csv(p)
        df = clean_df(df)
        df["day"] = day
        parts.append(df)


    all_df = pd.concat(parts, ignore_index=True)

    # Save as parquet for speed
    out_path = os.path.join(OUT_DIR, "cicids2017_all.parquet")
    all_df.to_parquet(out_path, index=False)
    print("Saved:", out_path)
    print(all_df[["day", "y"]].value_counts().head(20))

if __name__ == "__main__":
    main()
