"""Input loading and fold-fitted preprocessing for CICIDS2017."""

import glob
from pathlib import Path

import numpy as np
import pandas as pd


CAPTURE_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
PORT = "Destination Port"
NUMERIC_FEATURES = (
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Total Length of Bwd Packets',
    'Fwd Packet Length Max',
    'Fwd Packet Length Min',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Min',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Flow IAT Max',
    'Flow IAT Min',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Bwd IAT Total',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Bwd IAT Max',
    'Bwd IAT Min',
    'Fwd PSH Flags',
    'Bwd PSH Flags',
    'Fwd URG Flags',
    'Bwd URG Flags',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Min Packet Length',
    'Max Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'FIN Flag Count',
    'SYN Flag Count',
    'RST Flag Count',
    'PSH Flag Count',
    'ACK Flag Count',
    'URG Flag Count',
    'CWE Flag Count',
    'ECE Flag Count',
    'Down/Up Ratio',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Avg Bwd Segment Size',
    'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'act_data_pkt_fwd',
    'min_seg_size_forward',
    'Active Mean',
    'Active Std',
    'Active Max',
    'Active Min',
    'Idle Mean',
    'Idle Std',
    'Idle Max',
    'Idle Min',
)
FEATURE_COLUMNS = (PORT, *NUMERIC_FEATURES)
INPUT_COLUMNS = frozenset((*FEATURE_COLUMNS, "Label"))


def load_tabular_flows(folder="TrafficLabelling"):
    """Return raw feature rows, binary labels, and aligned capture metadata."""
    frames = []
    files = sorted(Path(folder).glob("*ISCX.csv"))
    source_names = [path.name for path in files]
    for file_index, path in enumerate(files):
        day = path.name.partition("-")[0]
        if day not in CAPTURE_DAYS:
            raise ValueError(f"Unknown capture day in {path.name}")
        frame = pd.read_csv(
            path,
            encoding="cp1252",
            usecols=lambda name: name.strip() in INPUT_COLUMNS,
            low_memory=False,
        )
        frame.columns = frame.columns.str.strip()
        missing = INPUT_COLUMNS - set(frame.columns)
        if missing or not frame.columns.is_unique:
            raise ValueError(f"{path.name}: invalid schema, missing {sorted(missing)}")
        labels = frame.pop("Label").astype("string").str.strip()
        if labels.isna().any() or labels.eq("").any():
            raise ValueError(f"{path.name}: missing or blank labels")
        frame["y"] = labels.ne("BENIGN").astype("int8")
        frame["capture_day"] = pd.Categorical.from_codes(
            np.full(len(frame), CAPTURE_DAYS.index(day), dtype=np.int8),
            categories=CAPTURE_DAYS,
        )
        frame["source_file"] = pd.Categorical.from_codes(
            np.full(len(frame), file_index, dtype=np.intp),
            categories=source_names,
        )
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No CICIDS2017 CSVs in {folder}")

    flows = pd.concat(frames, ignore_index=True)
    target = flows.pop("y")
    metadata = pd.DataFrame({
        "capture_day": flows.pop("capture_day"),
        "source_file": flows.pop("source_file"),
    })
    return flows, target, metadata


def day_splits(metadata):
    """Return expanding development splits and the final Friday holdout."""
    days = metadata["capture_day"]
    if set(days) != set(CAPTURE_DAYS):
        raise ValueError("Day splits require captures for all five weekdays")

    def split(next_day):
        earlier_days = CAPTURE_DAYS[:CAPTURE_DAYS.index(next_day)]
        train = np.flatnonzero(days.isin(earlier_days).to_numpy())
        held_out = np.flatnonzero(days.eq(next_day).to_numpy())
        return train, held_out

    validation = [split("Wednesday"), split("Thursday")]
    return validation, split("Friday")


def nonfinite_to_nan(values):
    """Run a fixed conversion before learning any imputation statistics."""
    numeric = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(numeric), numeric, np.nan)


def make_flow_pipeline(estimator, *, scale_numeric=False, encode_port=False):
    """Clone and fit this pipeline on raw feature rows in each training fold."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    numeric_steps = [
        ("finite", FunctionTransformer(nonfinite_to_nan)),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    branches = [("numeric", Pipeline(numeric_steps), list(NUMERIC_FEATURES))]
    if encode_port:
        port_steps = [
            ("finite", FunctionTransformer(nonfinite_to_nan)),
            ("impute", SimpleImputer(strategy="constant", fill_value=-1)),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
        branches.append(("port", Pipeline(port_steps), [PORT]))

    return Pipeline([
        ("preprocess", ColumnTransformer(branches, sparse_threshold=1.0)),
        ("model", estimator),
    ])


def load(folder):
    files = sorted(glob.glob(f"{folder}/**/*ISCX.csv", recursive=True))
    print(f"Found {len(files)} files")
    for f in files:
        print("  ", f)
    dfs = [pd.read_csv(f, encoding="cp1252", low_memory=False) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip()
    return df


if __name__ == "__main__":
    df = load(".")
    missing = {"Timestamp", "Source IP", "Destination IP"} - set(df.columns)
    if missing:
        raise SystemExit(
            f"Graph input is missing {sorted(missing)}. "
            "See the README for the tabular pipeline using TrafficLabelling."
        )
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
