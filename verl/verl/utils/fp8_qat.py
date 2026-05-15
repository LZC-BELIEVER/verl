"""FP8 W8A8 Quantization-Aware Training (QAT) utilities.

This module implements FP8 W8A8 fake-quantization for Megatron training,
designed to match exactly the quantization scheme used by vLLM FP8 rollout:
  - Weights: blockwise FP8 (e4m3fn, block_size=[128, 128])
  - Activations: per-token dynamic FP8 (matching activation_scheme="dynamic")

The implementation uses STE (Straight-Through Estimator): forward pass simulates
quantization error, backward pass passes gradients through unchanged.
"""

import re

import torch
import torch.nn.functional as F

from verl.utils.kernel.fp8_kernel import scaled_fp8_blockwise

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
WEIGHT_BLOCK_SIZE = [128, 128]


# ---------------------------------------------------------------------------
# STE: Blockwise FP8 fake-quantization for weights
# ---------------------------------------------------------------------------


def _blockwise_dequant(fp8_data: torch.Tensor, descale: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Dequantize blockwise FP8 tensor back to float32.

    Args:
        fp8_data: FP8 quantized tensor of shape (M, N).
        descale: Per-block descale factors of shape (blk_m, blk_n).
                 descale = amax / FP8_MAX, multiply to dequantize.
        shape: Original (M, N) shape before any padding.

    Returns:
        Float32 tensor of shape (M, N).
    """
    M, N = shape
    B = 128
    # Normalize descale: Triton returns (blk_m, blk_n), PyTorch returns (blk_m, blk_n, 1)
    if descale.dim() == 3:
        descale = descale.squeeze(-1)
    blk_m, blk_n = descale.shape

    # Pad fp8_data to block boundary if needed
    pad_m = blk_m * B - M
    pad_n = blk_n * B - N
    data_f32 = fp8_data.float()
    if pad_m > 0 or pad_n > 0:
        data_f32 = F.pad(data_f32, (0, pad_n, 0, pad_m))

    # Reshape to (blk_m, B, blk_n, B), broadcast-multiply descale, reshape back
    x = data_f32.reshape(blk_m, B, blk_n, B)
    s = descale.unsqueeze(1).unsqueeze(3)  # (blk_m, 1, blk_n, 1)
    result = (x * s).reshape(blk_m * B, blk_n * B)
    return result[:M, :N]


class FP8BlockwiseSTE(torch.autograd.Function):
    """STE fake-quantization: blockwise FP8 for weights.

    Forward: quantize weight to FP8 (blockwise, [128,128]) then dequantize back to input dtype.
    Backward: pass gradient straight through (STE).
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, block_size: list) -> torch.Tensor:
        fp8_w, descale = scaled_fp8_blockwise(weight.detach().float(), block_size)
        w_dq = _blockwise_dequant(fp8_w, descale, weight.shape)
        return w_dq.to(weight.dtype)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad, None  # STE: pass gradient through unchanged


# ---------------------------------------------------------------------------
# STE: Per-token dynamic FP8 fake-quantization for activations
# ---------------------------------------------------------------------------


class FP8PerTokenSTE(torch.autograd.Function):
    """STE fake-quantization: per-token dynamic FP8 for activations.

    Matches vLLM's activation_scheme="dynamic": each token is independently
    quantized to FP8 based on its own max absolute value.

    Forward: quantize each token to FP8 then dequantize back to input dtype.
    Backward: pass gradient straight through (STE).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        orig_dtype = x.dtype
        # Reshape to (T, H) for per-token treatment
        x2d = x.reshape(-1, orig_shape[-1]).float()
        amax = x2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)  # (T, 1)
        scale = FP8_MAX / amax  # quantization scale
        x_fp8 = (x2d * scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        x_dq = x_fp8.float() * (amax / FP8_MAX)  # dequantize
        return x_dq.to(orig_dtype).reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad  # STE: pass gradient through unchanged


# ---------------------------------------------------------------------------
# Hook helpers
# ---------------------------------------------------------------------------


def _should_ignore(name: str, patterns: list) -> bool:
    """Check if a module name matches any ignore pattern."""
    for p in patterns:
        if p.startswith("re:"):
            if re.search(p[3:], name):
                return True
        elif p in name:
            return True
    return False


def _make_pre_hook(block_size: list):
    """Create a forward pre-hook that fake-quantizes weight and activation."""

    def pre_hook(module, args, kwargs):
        orig_w = module.weight
        # Fake-quantize weight via STE
        w_fq = FP8BlockwiseSTE.apply(orig_w, block_size)
        # Temporarily swap the weight parameter so the TE linear sees the fake-quantized weight.
        # We use _parameters directly to avoid __setattr__ side-effects (e.g. re-registering
        # the parameter with a new name). Gradient flows: w_fq -> orig_w via STE.
        module._fp8qat_orig_weight = orig_w
        module._parameters["weight"] = torch.nn.Parameter(w_fq, requires_grad=orig_w.requires_grad)

        # Fake-quantize activation
        x = args[0] if args else kwargs.get("input_", None)
        if x is not None:
            x_fq = FP8PerTokenSTE.apply(x)
            if args:
                return (x_fq,) + args[1:], kwargs
            kwargs["input_"] = x_fq
            return args, kwargs

    return pre_hook


def _make_post_hook():
    """Create a forward hook that restores the original weight parameter."""

    def post_hook(module, input, output):
        orig = getattr(module, "_fp8qat_orig_weight", None)
        if orig is not None:
            module._parameters["weight"] = orig
            del module._fp8qat_orig_weight

    return post_hook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_fp8_w8a8_qat_hooks(modules: list, qat_config) -> list:
    """Register FP8 W8A8 fake-quant hooks on Megatron TE linear layers.

    Iterates through all module chunks and registers forward pre/post hooks on
    TEColumnParallelLinear and TERowParallelLinear instances (excluding layers
    matching qat_config.ignore_patterns).

    Args:
        modules: List of Megatron model chunk modules.
        qat_config: QATEngineConfig with ignore_patterns field.

    Returns:
        List of hook handles (can be used to remove hooks later).
    """
    try:
        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TERowParallelLinear,
        )
    except ImportError as e:
        raise ImportError(
            "FP8 W8A8 QAT requires Megatron with TransformerEngine extensions. "
            f"Failed to import TE linear layers: {e}"
        )

    ignore_patterns = list(getattr(qat_config, "ignore_patterns", None) or [])
    block_size = WEIGHT_BLOCK_SIZE
    handles = []
    hook_count = 0

    for module_chunk in modules:
        for name, mod in module_chunk.named_modules():
            if not isinstance(mod, (TEColumnParallelLinear, TERowParallelLinear)):
                continue
            if _should_ignore(name, ignore_patterns):
                continue
            h_pre = mod.register_forward_pre_hook(_make_pre_hook(block_size), with_kwargs=True)
            h_post = mod.register_forward_hook(_make_post_hook())
            handles.extend([h_pre, h_post])
            hook_count += 1

    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"FP8 W8A8 QAT: registered hooks on {hook_count} TE linear layers")
    return handles
