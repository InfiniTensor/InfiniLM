"""
Utility functions for DeepSeek MLA testing.
"""
import torch
import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MLAConfig:
    """
    Configuration for DeepSeek MLA (Multi-head Latent Attention).
    Based on DeepSeek-V3/R1 architecture.
    """
    # Model dimensions
    dim: int = 7168                    # Hidden dimension
    n_heads: int = 128                 # Number of attention heads
    
    # Low-rank compression dimensions
    q_lora_rank: int = 1536           # Query low-rank compression
    kv_lora_rank: int = 512           # KV low-rank compression
    
    # Head dimensions
    qk_nope_head_dim: int = 128       # Non-positional QK dimension per head
    qk_rope_head_dim: int = 64        # RoPE positional QK dimension per head
    v_head_dim: int = 128             # Value dimension per head
    
    # RoPE parameters
    max_seq_len: int = 4096 * 4
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0
    
    @property
    def qk_head_dim(self) -> int:
        """Total QK dimension per head."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim
    
    @property
    def n_local_heads(self) -> int:
        """Number of local heads (for single GPU)."""
        return self.n_heads


def torch_synchronize(device: str):
    """Synchronize the specified device."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "musa":
        torch.musa.synchronize()


def torch_empty_cache(device: str):
    """Empty cache for the specified device."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "musa":
        torch.musa.empty_cache()


def precompute_freqs_cis(args: MLAConfig, device: str = "cuda") -> torch.Tensor:
    """
    Precompute complex exponential values for rotary positional embeddings.
    
    Args:
        args: MLA configuration
        device: Target device
    
    Returns:
        torch.Tensor: Precomputed complex exponential values
    """
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis.to(device)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to the input tensor.
    
    Args:
        x: Input tensor [batch, seq_len, n_heads, head_dim]
        freqs_cis: Precomputed complex exponential values
    
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str = "tensor",
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> Tuple[bool, dict]:
    """
    Compare two tensors and return comparison statistics.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        name: Name for logging
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Tuple of (is_close, stats_dict)
    """
    if tensor1.shape != tensor2.shape:
        return False, {"error": f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"}
    
    # Move to same device and dtype for comparison
    t1 = tensor1.float().cpu()
    t2 = tensor2.float().cpu()
    
    abs_diff = torch.abs(t1 - t2)
    rel_diff = abs_diff / (torch.abs(t2) + 1e-8)
    
    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    
    stats = {
        "name": name,
        "shape": list(tensor1.shape),
        "is_close": is_close,
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
    }
    
    return is_close, stats


def print_comparison_result(stats: dict):
    """Print comparison statistics in a formatted way."""
    print(f"  [{stats['name']}] shape={stats['shape']}")
    print(f"    is_close: {stats['is_close']}")
    print(f"    max_abs_diff: {stats['max_abs_diff']:.6e}")
    print(f"    mean_abs_diff: {stats['mean_abs_diff']:.6e}")
    print(f"    max_rel_diff: {stats['max_rel_diff']:.6e}")
    print(f"    mean_rel_diff: {stats['mean_rel_diff']:.6e}")
