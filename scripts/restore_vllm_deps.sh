#!/usr/bin/env bash
# Run this AFTER calibrate_fp8_kv.py finishes to restore vllm-compatible compressed-tensors.
# vllm 0.11.2 requires compressed-tensors==0.12.2; llmcompressor upgrades it to 0.14.x.

set -e
echo "Restoring compressed-tensors==0.12.2 for vllm 0.11.2 compatibility..."
pip install "compressed-tensors==0.12.2" --no-deps 2>&1
echo "Done. Verifying vllm can import..."
python3 -c "import vllm; print(f'vllm {vllm.__version__} OK')"
