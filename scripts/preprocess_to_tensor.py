
# scripts/preprocess_to_tensor.py
import os
import numpy as np
import pandas as pd
from glob import glob

FEATURES = ["open", "high", "low", "close", "volume"]
DATA_DIR = "data/minute"
OUTPUT_FILE = "data/price_tensor.npy"

def load_symbol(filepath):
    df = pd.read_parquet(filepath)
    df = df.set_index("datetime").sort_index()
    df = df[FEATURES]
    df = df.resample("1min").ffill()  # ensure minute alignment
    return df

def main():
    files = sorted(glob(f"{DATA_DIR}/*.parquet"))
    all_dfs = [load_symbol(f) for f in files]
    
    # Align all by datetime index
    common_index = all_dfs[0].index
    for df in all_dfs:
        common_index = common_index.intersection(df.index)

    all_dfs = [df.loc[common_index] for df in all_dfs]
    stacked = np.stack([df.values for df in all_dfs], axis=1)  # shape (T, N, F)

    print("Final shape:", stacked.shape)
    np.save(OUTPUT_FILE, stacked)

if __name__ == "__main__":
    main()
