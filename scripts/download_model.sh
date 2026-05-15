#!/bin/bash

set -e 
set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

model_name="Qwen/Qwen3-8B"
base_model_name=$(basename "$model_name")  # Added quotes and $

huggingface-cli download \
  "$model_name" \
  --local-dir "/lanzichang1/models/$base_model_name"
