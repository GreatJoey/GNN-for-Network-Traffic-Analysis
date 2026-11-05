#!/bin/bash

echo "Running prep"
python3 prep.py
echo "Training the model"
python3 train.py
echo "Outputting graphs"
python3 graphs.py