import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from prep import (
    CAPTURE_DAYS, FEATURE_COLUMNS, PORT, day_splits, load_tabular_flows,
    make_flow_pipeline,
)


class FlowPipelineTests(unittest.TestCase):
    def test_loader_keeps_nonfinite_rows_and_separates_metadata(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "Tuesday-WorkingHours.pcap_ISCX.csv"
            with path.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    *(f" {name}" for name in FEATURE_COLUMNS),
                    " Fwd Header Length", " Label", " Source IP",
                ])
                for label in (" BENIGN ", "Attack"):
                    features = dict.fromkeys(FEATURE_COLUMNS, 0)
                    features["Flow Bytes/s"] = "Infinity"
                    writer.writerow([*features.values(), 0, label, "192.0.2.1"])
            X, y, metadata = load_tabular_flows(folder)

        self.assertEqual(tuple(X.columns), FEATURE_COLUMNS)
        self.assertEqual(y.tolist(), [0, 1])
        self.assertTrue(np.isposinf(X["Flow Bytes/s"]).all())
        self.assertEqual(metadata["capture_day"].tolist(), ["Tuesday", "Tuesday"])
        self.assertEqual(metadata["source_file"].tolist(), [path.name] * 2)
        self.assertTrue(X.index.equals(y.index) and X.index.equals(metadata.index))

    def test_preprocessing_statistics_come_from_training_rows(self):
        X = pd.DataFrame(0.0, index=range(6), columns=list(FEATURE_COLUMNS))
        X["Flow Duration"] = [1.0, 3.0, np.inf, np.nan, 1_000.0, -np.inf]
        X[PORT] = [80, 443, 80, 443, 1234, 1234]
        X["capture_day"] = ["Monday"] * 4 + ["Wednesday"] * 2
        X["Label"] = ["BENIGN", "Attack"] * 3
        y = [0, 1, 0, 1]

        pipeline = clone(make_flow_pipeline(
            LogisticRegression(), scale_numeric=True, encode_port=True
        ))
        pipeline.fit(X.iloc[:4], y)
        preprocess = pipeline.named_steps["preprocess"]
        numeric = preprocess.named_transformers_["numeric"]
        encoder = preprocess.named_transformers_["port"].named_steps["encode"]

        def check_training_statistics():
            self.assertEqual(numeric.named_steps["impute"].statistics_[0], 2.0)
            self.assertEqual(numeric.named_steps["scale"].mean_[0], 2.0)
            np.testing.assert_array_equal(encoder.categories_[0], [80, 443])

        check_training_statistics()
        self.assertEqual(len(pipeline.predict(X.iloc[4:])), 2)
        transformed = preprocess.transform(X.iloc[4:]).toarray()
        self.assertEqual(transformed.shape, (2, len(FEATURE_COLUMNS) + 1))
        self.assertTrue(np.isfinite(transformed).all())
        np.testing.assert_array_equal(transformed[:, -2:], 0)
        X.loc[4:, ["Flow Duration", PORT]] = -10_000.0
        pipeline.predict(X.iloc[4:])
        check_training_statistics()

    def test_day_splits_are_chronological_and_keep_capture_days_intact(self):
        metadata = pd.DataFrame({"capture_day": list(reversed(CAPTURE_DAYS)) * 2})
        validation, holdout = day_splits(metadata)
        for count, (train, held_out) in enumerate([*validation, holdout], start=2):
            self.assertEqual(
                set(metadata.iloc[train]["capture_day"]), set(CAPTURE_DAYS[:count])
            )
            self.assertEqual(set(metadata.iloc[held_out]["capture_day"]), {CAPTURE_DAYS[count]})
            self.assertEqual(len(held_out), 2)
            self.assertFalse(np.intersect1d(train, held_out).size)


if __name__ == "__main__":
    unittest.main()
