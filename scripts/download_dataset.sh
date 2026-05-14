#!/bin/bash

set -e
set -x

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download HuggingFaceH4/MATH-500 --repo-type dataset --local-dir "/lanzichang1/data/math2"