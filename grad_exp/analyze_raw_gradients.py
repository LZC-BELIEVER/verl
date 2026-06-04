#!/usr/bin/env python3
"""
Analyze raw gradients to understand why TIS has higher variance than Vanilla in Norm layers.

This script:
1. Loads raw gradients from bf16, fp8, fp8_vanilla
2. Focuses on layers with high variance in TIS (e.g., model.layers.2.self_attn.q_norm.weight)
3. Computes statistics to identify the root cause
"""

import torch
import numpy as np
from pathlib import Path


def load_gradient(path):
    """Load gradient dict from .pt file."""
    return torch.load(path, map_location="cpu")


def analyze_layer_gradient(grad_tensor, layer_name):
    """Compute statistics for a single layer's gradient."""
    grad_flat = grad_tensor.flatten().float()

    stats = {
        "layer": layer_name,
        "shape": tuple(grad_tensor.shape),
        "mean": grad_flat.mean().item(),
        "std": grad_flat.std().item(),
        "min": grad_flat.min().item(),
        "max": grad_flat.max().item(),
        "abs_mean": grad_flat.abs().mean().item(),
        "abs_max": grad_flat.abs().max().item(),
        "num_zeros": (grad_flat == 0).sum().item(),
        "num_elements": grad_flat.numel(),
    }

    # Percentiles
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for p in percentiles:
        stats[f"p{p}"] = torch.quantile(grad_flat.abs(), p / 100.0).item()

    return stats


def main():
    base_dir = Path("/lanzichang1/new_verl/grads/ori_grad")

    # Load raw gradients
    bf16_grad = load_gradient(base_dir / "bf16" / "grad_raw.pt")
    tis_grad = load_gradient(base_dir / "fp8" / "grad_raw.pt")
    vanilla_grad = load_gradient(base_dir / "fp8_vanilla" / "grad_raw.pt")

    print("=" * 80)
    print("Raw Gradient Analysis: TIS vs Vanilla Variance")
    print("=" * 80)
    print()

    # Focus on problematic layers from report_tis.md
    problematic_layers = [
        "model.layers.2.self_attn.q_norm.weight",
        "model.layers.2.self_attn.k_norm.weight",
        "model.layers.5.self_attn.q_norm.weight",
        "model.layers.5.self_attn.k_norm.weight",
        "model.layers.8.self_attn.q_norm.weight",
        "model.layers.8.self_attn.k_norm.weight",
    ]

    # Also check some stable layers for comparison
    stable_layers = [
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.2.self_attn.k_proj.weight",
        "model.layers.2.mlp.gate_proj.weight",
    ]

    all_layers = problematic_layers + stable_layers

    for layer_name in all_layers:
        if layer_name not in bf16_grad or layer_name not in tis_grad or layer_name not in vanilla_grad:
            print(f"⚠ Layer {layer_name} not found in all gradient files")
            continue

        print(f"\n{'=' * 80}")
        print(f"Layer: {layer_name}")
        print(f"{'=' * 80}")

        bf16_stats = analyze_layer_gradient(bf16_grad[layer_name], "BF16")
        tis_stats = analyze_layer_gradient(tis_grad[layer_name], "TIS")
        vanilla_stats = analyze_layer_gradient(vanilla_grad[layer_name], "Vanilla")

        # Print comparison table
        print(f"\n{'Metric':<20} {'BF16':>15} {'TIS':>15} {'Vanilla':>15}")
        print("-" * 68)

        metrics = ["mean", "std", "abs_mean", "abs_max", "min", "max"]
        for metric in metrics:
            print(f"{metric:<20} {bf16_stats[metric]:>15.6e} {tis_stats[metric]:>15.6e} {vanilla_stats[metric]:>15.6e}")

        print()
        print("Percentiles (absolute value):")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            key = f"p{p}"
            print(f"  {key:<18} {bf16_stats[key]:>15.6e} {tis_stats[key]:>15.6e} {vanilla_stats[key]:>15.6e}")

        # Compute relative differences
        print()
        print("Relative to BF16:")
        print(f"  TIS std ratio:     {tis_stats['std'] / bf16_stats['std']:.4f}x")
        print(f"  Vanilla std ratio: {vanilla_stats['std'] / bf16_stats['std']:.4f}x")
        print(f"  TIS abs_max ratio: {tis_stats['abs_max'] / bf16_stats['abs_max']:.4f}x")
        print(f"  Vanilla abs_max ratio: {vanilla_stats['abs_max'] / bf16_stats['abs_max']:.4f}x")

        # Check for outliers
        bf16_flat = bf16_grad[layer_name].flatten().float()
        tis_flat = tis_grad[layer_name].flatten().float()
        vanilla_flat = vanilla_grad[layer_name].flatten().float()

        # Count extreme values (> 3 std from mean)
        bf16_outliers = (bf16_flat.abs() > bf16_stats['mean'] + 3 * bf16_stats['std']).sum().item()
        tis_outliers = (tis_flat.abs() > tis_stats['mean'] + 3 * tis_stats['std']).sum().item()
        vanilla_outliers = (vanilla_flat.abs() > vanilla_stats['mean'] + 3 * vanilla_stats['std']).sum().item()

        print()
        print(f"Outliers (>3σ from mean):")
        print(f"  BF16:    {bf16_outliers:>6} / {bf16_stats['num_elements']} ({100*bf16_outliers/bf16_stats['num_elements']:.2f}%)")
        print(f"  TIS:     {tis_outliers:>6} / {tis_stats['num_elements']} ({100*tis_outliers/tis_stats['num_elements']:.2f}%)")
        print(f"  Vanilla: {vanilla_outliers:>6} / {vanilla_stats['num_elements']} ({100*vanilla_outliers/vanilla_stats['num_elements']:.2f}%)")

    print("\n" + "=" * 80)
    print("Analysis complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
