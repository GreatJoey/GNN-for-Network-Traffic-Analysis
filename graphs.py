import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

df = pd.read_parquet("flows.parquet")

# Makes it record IPs and Source Ports
drop = ["Flow ID", "Source IP", "Destination IP", "Source Port", "Timestamp", "Label", "y"]
feat_cols = [c for c in df.select_dtypes("number").columns if c not in drop]

# cleans the blank data and some other stuff
df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

df["window"] = df["Timestamp"].dt.floor("5min")
windows = sorted(df["window"].unique())

day = pd.Series(windows).dt.date
train_w, test_w = [], []
for d, ws in pd.Series(windows).groupby(day.values):
    k = int(len(ws) * 0.7)
    train_w += list(ws.iloc[:k])
    test_w += list(ws.iloc[k:])

# standardize using train stats to prevent leakage
tr = df[df["window"].isin(train_w)]
mu, sd = tr[feat_cols].mean(), tr[feat_cols].std().replace(0, 1)
df[feat_cols] = ((df[feat_cols] - mu) / sd).clip(-10, 10)

def make_graph(g):
    ids, uniq = pd.factorize(pd.concat([g["Source IP"], g["Destination IP"]]))
    n = len(g)
    src, dst = ids[:n], ids[n:]
    edge_attr = torch.tensor(g[feat_cols].values, dtype=torch.float)
    return Data(
        x=torch.ones(len(uniq), edge_attr.size(1)), 
        edge_index=torch.tensor(np.vstack([src, dst]), dtype=torch.long),
        edge_attr=edge_attr,
        y=torch.tensor(g["y"].values, dtype=torch.long),
    )

groups = dict(tuple(df.groupby("window")))
train_set = [make_graph(groups[w]) for w in train_w]
test_set = [make_graph(groups[w]) for w in test_w]

if __name__ == "__main__":
    # baseline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    trd = df[df["window"].isin(train_w)]
    ted = df[df["window"].isin(test_w)]
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced")
    rf.fit(trd[feat_cols], trd["y"])
    print(classification_report(ted["y"], rf.predict(ted[feat_cols]), digits=4))