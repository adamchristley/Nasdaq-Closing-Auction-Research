#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
kaggle competitions download -c optiver-trading-at-the-close -p data
unzip -o data/optiver-trading-at-the-close.zip -d data
printf 'Dataset extracted under data/. Raw data is ignored by git.\n'
