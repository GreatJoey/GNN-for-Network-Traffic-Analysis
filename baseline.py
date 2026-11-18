"""
Gradient boosting baseline for CIC-IDS2017 (flow-level, binary attack vs benign).

Unit of independence = 5-minute window (never individual flows).

Modes
  python baseline.py              blocked CV with a purge gap on the dev set (primary)
  python baseline.py --lodo       leave-one-day-out on the dev set (stress test)
  python baseline.py --final      fit on all dev data, score the held-out test ONCE

Every run appends a row to experiments.csv and saves per-fold results to results/.

Assumes prep.py has produced flows.parquet with a *correct* Timestamp
(see the AM/PM fix) plus the original `Label` column and the binary `y`.
"""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# ----------------------------------------------------------------- config
PARQUET = "flows.parquet"
LOG = Path("experiments.csv")
OUT = Path("results")
SEED = 42
WINDOW = "5min"
TEST_FRAC = 0.15      # last 15% of each day's windows -> held-out test
GAP = 1               # windows purged on each side of any boundary (5 min)
N_FOLDS = 5
N_BLOCKS = 25         # contiguous time blocks dealt round-robin into folds
THR = 0.5             # threshold for F1 / per-type recall
FPR_TARGET = 0.01     # operating point for "recall at fixed false-alarm rate"

DROP = ["Flow ID", "Source IP", "Destination IP", "Source Port", "Destination Port",
        "Timestamp", "Label", "y", "window"]
LOG_COLS = ["date", "who", "script", "model", "features", "split", "hyperparameters",
            "seed", "pr_auc", "f1", "recall_at_1pct_fpr", "per_type_recall", "notes"]


# ------------------------------------------------------------ pipeline
def inf_to_nan(X):
    X = np.array(X, dtype=np.float32)          # copy, never mutate the caller's data
    return np.where(np.isfinite(X), X, np.nan)


def build_pipeline(max_iter, lr, leaves):
    """All preprocessing lives inside the Pipeline, so it is refit on each training fold.
    No scaler: trees are scale-invariant, so it would add nothing here."""
    return Pipeline([
        ("inf_to_nan", FunctionTransformer(inf_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=lr, max_leaf_nodes=leaves,
            class_weight="balanced", early_stopping=False, random_state=SEED)),
    ])


# ------------------------------------------------------------ data + roles
def load():
    df = pd.read_parquet(PARQUET)
    df["Label"] = df["Label"].str.strip()

    # CIC-IDS2017 timestamps have no AM/PM: hours 1-7 are really afternoon.
    # Fixed here (not in prep.py) so the original pipeline files stay untouched.
    # Safe to rerun: if the parquet was already fixed, no hours are < 8 and nothing changes.
    early = df["Timestamp"].dt.hour < 8
    df.loc[early, "Timestamp"] += pd.Timedelta(hours=12)
    print(f"AM/PM fix applied to {int(early.sum()):,} flows")
    print(f"Time range after fix: {df['Timestamp'].min()} -> {df['Timestamp'].max()}")
    print("Flows per day:")
    print(df["Timestamp"].dt.date.value_counts().sort_index().to_string())

    feat_cols = [c for c in df.select_dtypes("number").columns
                 if c not in DROP and not c.endswith(".1")]   # drops 'Fwd Header Length.1'
    df["window"] = df["Timestamp"].dt.floor(WINDOW)
    wins = np.array(sorted(df["window"].unique()))
    wpos = pd.Series(np.arange(len(wins)), index=wins)
    df["wpos"] = df["window"].map(wpos).to_numpy()
    days = pd.Series(wins).dt.date.to_numpy()

    # Role of each window: dev / gap (purged buffer) / test
    role = np.full(len(wins), "dev", dtype=object)
    for d in pd.unique(days):
        idx = np.where(days == d)[0]
        k = int(len(idx) * (1 - TEST_FRAC))
        role[idx[k:]] = "test"
        role[idx[max(k - GAP, 0):k]] = "gap"
    return df, feat_cols, wins, days, role


def make_blocked_folds(n_windows):
    """Contiguous time blocks dealt round-robin into folds; training windows
    within GAP windows of a validation window are purged."""
    block = np.minimum(np.arange(n_windows) * N_BLOCKS // n_windows, N_BLOCKS - 1)
    fold = block % N_FOLDS
    for f in range(N_FOLDS):
        val = fold == f
        near = np.convolve(val.astype(int), np.ones(2 * GAP + 1), mode="same") > 0
        yield f"block{f}", ~val & ~near, val


def make_lodo_folds(win_day):
    for d in pd.unique(win_day):
        val = win_day == d
        yield f"day{d}", ~val, val


# ----------------------------------------------------------------- metrics
def evaluate(y, p, labels):
    fpr, tpr, _ = roc_curve(y, p)
    per_type = {}
    for lab in np.unique(labels):
        if lab != "BENIGN":
            m = labels == lab
            per_type[lab] = float((p[m] >= THR).mean())
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, p >= THR)),
        "recall_at_fpr": float(tpr[fpr <= FPR_TARGET].max()),
        "per_type": per_type,
        "n_val": int(len(y)), "n_attack": int(y.sum()),
    }


def log_run(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow(row)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--lodo", action="store_true", help="leave-one-day-out on dev set")
    mode.add_argument("--final", action="store_true", help="fit on dev, score test once")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--quick", action="store_true",
                    help="train on a random 20%% of each training fold (debugging only)")
    ap.add_argument("--who", default="")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    df, feat_cols, wins, days, role = load()
    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    y_all = df["y"].to_numpy()
    lab_all = df["Label"].to_numpy()
    row_role = role[df["wpos"].to_numpy()]

    print(f"{len(feat_cols)} features, {len(df):,} flows, {len(wins)} windows")
    print("\nLabel counts by role (check every attack type appears in dev):")
    print(pd.crosstab(lab_all, row_role).reindex(columns=["dev", "gap", "test"], fill_value=0))

    pipe = build_pipeline(args.max_iter, args.lr, args.leaves)
    rng = np.random.default_rng(SEED)
    hp = {"max_iter": args.max_iter, "learning_rate": args.lr,
          "max_leaf_nodes": args.leaves, "class_weight": "balanced", "quick": args.quick}

    dev_rows = np.where(row_role == "dev")[0]
    dev_w = np.where(role == "dev")[0]
    local = np.searchsorted(dev_w, df["wpos"].to_numpy()[dev_rows])   # dev-window index per dev row

    if args.final:
        split_name, folds = "final_test", None
    elif args.lodo:
        split_name = "leave_one_day_out"
        folds = list(make_lodo_folds(pd.Series(days[dev_w]).to_numpy()))
    else:
        split_name = f"blocked_cv_{N_FOLDS}fold_gap{GAP}"
        folds = list(make_blocked_folds(len(dev_w)))

    def fit_predict(tr_rows, va_rows):
        if args.quick:
            tr_rows = rng.choice(tr_rows, size=int(len(tr_rows) * 0.2), replace=False)
        model = clone(pipe).fit(X_all[tr_rows], y_all[tr_rows])
        return model.predict_proba(X_all[va_rows])[:, 1]

    results = []
    if args.final:
        te_rows = np.where(row_role == "test")[0]
        p = fit_predict(dev_rows, te_rows)
        r = evaluate(y_all[te_rows], p, lab_all[te_rows])
        r["fold"] = "test"
        results.append(r)
        print(f"\nTEST  PR-AUC {r['pr_auc']:.4f} | F1 {r['f1']:.4f} | "
              f"recall@{FPR_TARGET:.0%}FPR {r['recall_at_fpr']:.4f}")
    else:
        for name, tr_w, va_w in folds:
            tr_rows = dev_rows[tr_w[local]]
            va_rows = dev_rows[va_w[local]]
            if y_all[va_rows].sum() == 0 or y_all[va_rows].sum() == len(va_rows):
                print(f"{name}: skipped (validation has only one class)")
                continue
            p = fit_predict(tr_rows, va_rows)
            r = evaluate(y_all[va_rows], p, lab_all[va_rows])
            r["fold"] = name
            results.append(r)
            print(f"{name:>14} | PR-AUC {r['pr_auc']:.4f} | F1 {r['f1']:.4f} | "
                  f"recall@{FPR_TARGET:.0%}FPR {r['recall_at_fpr']:.4f} | "
                  f"n={r['n_val']:,} attacks={r['n_attack']:,}")

    # ---- aggregate, save, log
    flat = pd.DataFrame([{k: v for k, v in r.items() if k != "per_type"} for r in results])
    flat.to_csv(OUT / f"gb_{split_name}_folds.csv", index=False)
    pt = pd.DataFrame([r["per_type"] for r in results], index=[r["fold"] for r in results])
    pt.to_csv(OUT / f"gb_{split_name}_per_type_recall.csv")
    pt_mean = pt.mean(skipna=True).round(4).to_dict()

    def ms(col):
        return f"{flat[col].mean():.4f}+-{flat[col].std(ddof=0):.4f}"

    print(f"\nMEAN +- STD over {len(flat)} fold(s)")
    print(f"  PR-AUC {ms('pr_auc')} | F1 {ms('f1')} | recall@{FPR_TARGET:.0%}FPR {ms('recall_at_fpr')}")
    print("  per-type recall (mean over folds where the type appears):")
    for k, v in pt_mean.items():
        print(f"    {k:<30} {v:.4f}")

    log_run({
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "who": args.who, "script": "baseline.py", "model": "HistGradientBoosting",
        "features": f"{len(feat_cols)} flow feats (no IPs/ports)", "split": split_name,
        "hyperparameters": json.dumps(hp), "seed": SEED,
        "pr_auc": ms("pr_auc"), "f1": ms("f1"), "recall_at_1pct_fpr": ms("recall_at_fpr"),
        "per_type_recall": json.dumps(pt_mean), "notes": args.notes,
    })


if __name__ == "__main__":
    main()