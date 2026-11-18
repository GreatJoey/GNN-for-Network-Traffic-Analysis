# GNN-for-Network-Traffic-Analysis

## Part 3 preprocessing

The CSVs in `TrafficLabelling` support the tabular baseline. The graph
workflow below requires an export with flow timestamps and endpoint IPs.
Install the tabular dependencies with
`python -m pip install pandas numpy "scikit-learn>=1.2"`.

```python
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from prep import day_splits, load_tabular_flows, make_flow_pipeline

X, y, metadata = load_tabular_flows()
validation, final_holdout = day_splits(metadata)
pipeline = make_flow_pipeline(LogisticRegression(), scale_numeric=True)

for train, valid in validation:
    fitted = clone(pipeline).fit(X.iloc[train], y.iloc[train])
    attack_scores = fitted.predict_proba(X.iloc[valid])[:, 1]
    # Score against y.iloc[valid] using the baseline evaluation metrics.
```

The development splits validate on Wednesday and Thursday in turn, using
only earlier days for training. Reserve all Friday captures for a single
final evaluation after fixing model and preprocessing choices. These splits
represent later captures within the same lab environment.

The loader returned 2,830,743 aligned rows across the eight files, and both
classes were present in every training fold. Hashing all 77 parsed feature
columns, including destination port, flagged these held-out rows with a
matching feature vector in an earlier capture.

| Held-out day | Training flows | Held-out flows | Matching vectors |
| --- | ---: | ---: | ---: |
| Wednesday | 975,827 | 692,703 | 21,357 |
| Thursday | 1,668,530 | 458,968 | 30,191 |
| Friday | 2,127,498 | 703,245 | 36,151 |

The CSVs lack flow IDs and endpoint metadata, so matching vectors alone
cannot identify duplicate observations. The loader retains these rows.

The loader retains flow rows with missing or infinite numeric values. It
uses an explicit feature list and keeps capture day and filename in separate
metadata. It uses the first `Fwd Header Length` column because the two
copies in the supplied files are identical. Feature selection is currently
disabled. Destination port is excluded by default to reduce dependence
on which services were targeted in a capture. Set `encode_port=True`
to evaluate it as a categorical feature with a training-fitted vocabulary
that ignores unseen ports. Use `scale_numeric=False` for a random forest.

Run `python -m unittest test_prep` to check the chronological splits,
training-only statistics, unseen ports, and metadata exclusion.

## Graph workflow

**Be sure to download the dataset here**
https://www.unb.ca/cic/datasets/ids-2017.html

**How to run**

```source setup.sh```

```./run.sh```

**Output should look something like this**
```
2017-07-04 01:00:00 2017-07-07 12:59:00
Label
BENIGN                        1741672
DoS Hulk                       230123
PortScan                       158804
DDoS                           128025
DoS GoldenEye                   10293
FTP-Patator                      7935
SSH-Patator                      5897
DoS slowloris                    5796
DoS Slowhttptest                 5499
Bot                              1956
Web Attack – Brute Force         1507
Web Attack – XSS                  652
Infiltration                       36
Web Attack – Sql Injection         21
Heartbleed                         11
Name: count, dtype: int64
Training the model
epoch 1 | F1 0.6349 | PR-AUC 0.6726
epoch 2 | F1 0.9423 | PR-AUC 0.9224
epoch 3 | F1 0.9426 | PR-AUC 0.9247
epoch 4 | F1 0.9381 | PR-AUC 0.9262
epoch 5 | F1 0.6975 | PR-AUC 0.6587

Outputting graphs
              precision    recall  f1-score   support

           0     0.6249    1.0000    0.7692    400610
           1     0.9962    0.0192    0.0377    245136

    accuracy                         0.6276    645746
   macro avg     0.8106    0.5096    0.4034    645746
weighted avg     0.7659    0.6276    0.4915    645746
```
