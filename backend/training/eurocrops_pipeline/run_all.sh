#!/bin/bash
set -e
cd "$(dirname "$0")"
PY=../../.venv/bin/python

echo "=== $(date) : starting EuroCrops fetch (18 countries) ==="
$PY run_fetch.py

echo "=== $(date) : starting BreizhCrops fetch ==="
$PY run_breizhcrops.py

echo "=== $(date) : starting training (combined, BreizhCrops folded in) ==="
$PY run_train.py --breizhcrops --epochs 20

echo "=== $(date) : ALL DONE ==="
