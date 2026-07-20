#!/bin/sh
# FLARE-AutoMSC Docker submission entry point.
# Reads cases from /workspace/inputs and writes seg masks + results.csv to /workspace/outputs.
set -e

python flare_predict.py \
  --input_dir /workspace/inputs \
  --output_dir /workspace/outputs \
  --model_dir /workspace/model_weights \
  --device cuda
