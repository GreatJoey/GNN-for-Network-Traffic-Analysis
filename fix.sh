#!/bin/bash
set -e
python prep.py
python baseline.py
python train.py