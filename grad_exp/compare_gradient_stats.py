#!/usr/bin/env python3
"""
梯度统计分析脚本：对比低精度训练方法与 BF16 基线的梯度误差

计算三种统计学指标：
1. L2 Distance of Means (均值的 L2 / 欧氏距离)
2. Cosine Similarity of Means (均值的余弦相似度)
3. Relative Noise Expansion (相对噪声膨胀率)
"""

import argparse
import torch
from pathlib import Path
from typing import Dict, Tuple
import warnings


def load_gradient_stats(data_dir: str) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    从指定目录加载梯度统计数据（均值和方差）

    Args:
        data_dir: 包含 grad_mean.pt 和 grad_var.pt 的目录路径

    Returns:
        字典，格式为 {layer_name: {"mean": tensor, "var": tensor}}
    """
    data_path = Path(data_dir)
    mean_file = data_path / "grad_mean.pt"
    var_file = data_path / "grad_var.pt"

    if not mean_file.exists() or not var_file.exists():
        raise FileNotFoundError(
            f"Missing gradient statistics files in {data_dir}. "
            f"Expected: grad_mean.pt and grad_var.pt"
        )

    mean_dict = torch.load(mean_file, map_location='cpu')
    var_dict = torch.load(var_file, map_location='cpu')

    # 合并为统一格式
    stats = {}
    for layer_name in mean_dict.keys():
        if layer_name not in var_dict:
            warnings.warn(f"Layer {layer_name} missing variance data, skipping")
            continue
        stats[layer_name] = {
            "mean": mean_dict[layer_name],
            "var": var_dict[layer_name]
        }

    return stats


def compute_l2_distance(
    mean_base: torch.Tensor,
    mean_target: torch.Tensor,
) -> float:
    """
    计算均值的 L2 距离（欧氏距离）

    公式: L2 = ||μ_base - μ_target||_2 = sqrt(Σ (μ_base - μ_target)^2)

    Args:
        mean_base: 基线均值张量
        mean_target: 目标均值张量

    Returns:
        标量，表示均值向量之间的 L2 距离
    """
    diff = (mean_base - mean_target).flatten()
    return torch.norm(diff, p=2).item()


def compute_cosine_similarity(
    mean_base: torch.Tensor,
    mean_target: torch.Tensor,
    eps: float = 1e-25
) -> float:
    """
    计算均值的余弦相似度

    公式: Sim = (μ_base · μ_target) / (||μ_base||_2 ||μ_target||_2)

    Args:
        mean_base: 基线均值张量
        mean_target: 目标均值张量
        eps: 数值稳定性的极小值

    Returns:
        标量，表示余弦相似度 [-1, 1]
    """
    # 展平为一维向量
    vec_base = mean_base.flatten()
    vec_target = mean_target.flatten()

    # 计算点积
    dot_product = torch.dot(vec_base, vec_target).item()

    # 计算范数
    norm_base = torch.norm(vec_base, p=2).item()
    norm_target = torch.norm(vec_target, p=2).item()

    # 处理除零异常
    if norm_base < eps or norm_target < eps:
        warnings.warn("Zero norm detected in cosine similarity, returning 1.0")
        return 1.0

    return dot_product / (norm_base * norm_target)


def compute_relative_noise_expansion(
    var_base: torch.Tensor,
    var_target: torch.Tensor,
    eps: float = 1e-25
) -> float:
    """
    计算相对噪声膨胀率

    公式: Δ_noise = (σ²_target - σ²_base) / σ²_base × 100%

    Args:
        var_base: 基线方差张量
        var_target: 目标方差张量
        eps: 数值稳定性的极小值

    Returns:
        标量，表示相对噪声膨胀率（百分比）
    """
    # 计算方差的平均值
    avg_var_base = torch.mean(var_base).item()
    avg_var_target = torch.mean(var_target).item()

    # 添加极小值保护，防止除零
    if avg_var_base < eps:
        warnings.warn(f"Baseline variance too small ({avg_var_base}), using eps={eps}")
        avg_var_base = eps

    # 计算相对变化率
    relative_change = ((avg_var_target - avg_var_base) / avg_var_base) * 100.0

    return relative_change


def compute_layer_metrics(
    baseline_stats: Dict[str, Dict[str, torch.Tensor]],
    target_stats: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, Dict[str, float]]:
    """
    逐层计算三种统计指标

    Args:
        baseline_stats: 基线梯度统计数据
        target_stats: 目标梯度统计数据

    Returns:
        字典，格式为 {layer_name: {"l2": float, "cosine": float, "noise": float}}
    """
    # 找到共有的层名
    common_layers = set(baseline_stats.keys()) & set(target_stats.keys())

    if len(common_layers) == 0:
        raise ValueError("No common layers found between baseline and target")

    results = {}

    for layer_name in sorted(common_layers):
        base_mean = baseline_stats[layer_name]["mean"]
        base_var = baseline_stats[layer_name]["var"]
        target_mean = target_stats[layer_name]["mean"]
        target_var = target_stats[layer_name]["var"]

        # 检查张量形状是否匹配
        if base_mean.shape != target_mean.shape or base_var.shape != target_var.shape:
            warnings.warn(
                f"Shape mismatch for layer {layer_name}, skipping. "
                f"Base: {base_mean.shape}, Target: {target_mean.shape}"
            )
            continue

        try:
            l2_dist = compute_l2_distance(base_mean, target_mean)
            cosine_sim = compute_cosine_similarity(base_mean, target_mean)
            noise_exp = compute_relative_noise_expansion(base_var, target_var)

            results[layer_name] = {
                "l2": l2_dist,
                "cosine": cosine_sim,
                "noise": noise_exp
            }
        except Exception as e:
            warnings.warn(f"Error computing metrics for layer {layer_name}: {e}")
            continue

    return results


def generate_markdown_report(
    results: Dict[str, Dict[str, float]],
    method_name: str,
    output_path: str
) -> None:
    """
    生成格式化的 Markdown 报告

    Args:
        results: 逐层计算的指标结果
        method_name: 目标方法名称（如 TIS, Vanilla）
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write(f"# 梯度统计分析报告: {method_name} vs Baseline (BF16)\n\n")
        f.write(f"**评估对象**: `{method_name}`  \n")
        f.write(f"**基线模型**: `BF16`\n\n")

        # 写入表格头
        f.write("## 逐层对比结果\n\n")
        f.write("| Layer Name | L2 Dist (Means) | Cosine Sim (Means) | Relative Noise Exp (%) |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")

        # 写入每一层的结果
        for layer_name in sorted(results.keys()):
            metrics = results[layer_name]
            l2 = metrics["l2"]
            cosine = metrics["cosine"]
            noise = metrics["noise"]

            # 格式化噪声膨胀率（带正负号）
            noise_str = f"{noise:+.1f}%" if noise >= 0 else f"{noise:.1f}%"

            # 使用科学计数法显示 L2 距离
            f.write(f"| {layer_name} | {l2:.4e} | {cosine:.4f} | {noise_str} |\n")

        f.write("\n")
        f.write("---\n")
        f.write(f"**总层数**: {len(results)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="评估低精度训练方法相对于 BF16 基线的梯度误差"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="高精度基线数据目录路径（包含 grad_mean.pt 和 grad_var.pt）"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="低精度目标数据目录路径（包含 grad_mean.pt 和 grad_var.pt）"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="目标方法的名称字符串（如 TIS 或 Vanilla），用于报告标题"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出的 Markdown 文件路径（如 report_tis.md）"
    )

    args = parser.parse_args()

    print(f"Loading baseline data from: {args.baseline}")
    baseline_stats = load_gradient_stats(args.baseline)
    print(f"  Loaded {len(baseline_stats)} layers")

    print(f"Loading target data from: {args.target}")
    target_stats = load_gradient_stats(args.target)
    print(f"  Loaded {len(target_stats)} layers")

    print("Computing layer-wise metrics...")
    results = compute_layer_metrics(baseline_stats, target_stats)
    print(f"  Computed metrics for {len(results)} layers")

    print(f"Generating Markdown report: {args.output}")
    generate_markdown_report(results, args.method, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
