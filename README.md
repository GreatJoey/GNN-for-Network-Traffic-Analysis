# GNN-for-Network-Traffic-Analysis

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
