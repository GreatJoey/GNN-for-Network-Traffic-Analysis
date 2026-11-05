import glob 
import numpy as np 
import pandas as pd

def load(folder):
    files = sorted(glob.glob(f"{folder}/**/*ISCX.csv", recursive=True))
    print(f"Found {len(files)} files")
    for f in files:
        print("  ", f)
    dfs = [pd.read_csv(f, encoding="cp1252", low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df

df = load(".")
df = df.drop_duplicates()
df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Timestamp"])
print(df["Timestamp"].min(), df["Timestamp"].max())

num_cols = df.select_dtypes("number").columns.difference(["Source Port"])
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=num_cols)

df["y"] = (df["Label"].str.strip() != "BENIGN").astype(int)
print(df["Label"].value_counts())
df.to_parquet("flows.parquet")
# make sure to pip install pyarrow