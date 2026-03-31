"""
MLA (Multi-head Latent Attention) Reference Implementation.

This module provides a PyTorch reference implementation of MLA based on
DeepSeek-V3/R1 architecture. It implements the "absorb" mode with 
kv_cache + pe_cache caching strategy.

Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
"""
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

# Add the current directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import MLAConfig, apply_rotary_emb, precompute_freqs_cis


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        norm = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).type_as(x)


class Linear(nn.Module):
    """
    Standard linear layer for BF16 computation.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.
    
    Implements the absorb mode with kv_cache + pe_cache caching strategy.
    This is more memory efficient than the naive approach as it stores:
    - kv_cache: compressed KV states (kv_lora_rank dimension)
    - pe_cache: positional encoding states (qk_rope_head_dim dimension)
    
    Instead of storing full K and V tensors.
    
    Attributes:
        dim: Input/output dimensionality
        n_heads: Number of attention heads
        n_local_heads: Number of local heads (for single GPU)
        q_lora_rank: Low-rank dimension for query projection
        kv_lora_rank: Low-rank dimension for key/value projection
        qk_nope_head_dim: Dimension for non-positional query/key projections
        qk_rope_head_dim: Dimension for rotary positional query/key projections
        qk_head_dim: Total query/key dimension per head
        v_head_dim: Value dimension per head
        softmax_scale: Scaling factor for attention scores
    """
    
    def __init__(self, args: MLAConfig):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_local_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        
        # Query projection (with LoRA)
        if self.q_lora_rank == 0:
            self.wq = Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        
        # KV projection (with LoRA)
        # Output: kv_lora_rank (for compressed KV) + qk_rope_head_dim (for PE)
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        
        # KV decompression projection
        # Output: n_heads * (qk_nope_head_dim + v_head_dim)
        self.wkv_b = Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        
        # Output projection
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim)
        
        # Attention scaling
        self.softmax_scale = self.qk_head_dim ** -0.5
        
        # Extended sequence length scaling
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        
        # Cache buffers (will be initialized per batch)
        self.kv_cache: Optional[torch.Tensor] = None  # [batch, max_seq, kv_lora_rank]
        self.pe_cache: Optional[torch.Tensor] = None  # [batch, max_seq, qk_rope_head_dim]
    
    def init_cache(self, batch_size: int, max_seq_len: int, device: str, dtype: torch.dtype):
        """Initialize KV and PE caches for a given batch size."""
        self.kv_cache = torch.zeros(
            batch_size, max_seq_len, self.kv_lora_rank,
            device=device, dtype=dtype
        )
        self.pe_cache = torch.zeros(
            batch_size, max_seq_len, self.qk_rope_head_dim,
            device=device, dtype=dtype
        )
    
    def update_cache(
        self, 
        kv: torch.Tensor, 
        k_pe: torch.Tensor, 
        start_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update KV and PE caches with new values.
        
        Args:
            kv: Compressed KV tensor [batch, seq_len, kv_lora_rank]
            k_pe: Position-encoded K tensor [batch, seq_len, qk_rope_head_dim]
            start_pos: Starting position in the sequence
        
        Returns:
            Tuple of (kv_cache[:, :end_pos], pe_cache[:, :end_pos])
        """
        bsz, seqlen, _ = kv.size()
        end_pos = start_pos + seqlen
        
        self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
        self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
        
        return self.kv_cache[:bsz, :end_pos], self.pe_cache[:bsz, :end_pos]
    
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for MLA with absorb mode.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            start_pos: Starting position for caching
            freqs_cis: Precomputed rotary embedding frequencies
            mask: Optional attention mask
        
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # ----- Query Projection -----
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        # Reshape to [batch, seq, n_heads, qk_head_dim]
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        
        # Split into non-positional and positional parts
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        
        # Apply RoPE to positional query
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # ----- KV Projection -----
        kv = self.wkv_a(x)
        
        # Split: kv_lora_rank for compressed KV, qk_rope_head_dim for PE
        kv, k_pe = torch.split(
            kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        
        # Apply RoPE to positional key (unsqueeze for head dim)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # ----- Absorb Mode Attention -----
        # Get wkv_b weight for absorption
        # wkv_b.weight: [n_heads * (qk_nope + v), kv_lora_rank]
        wkv_b = self.wkv_b.weight  # [n_heads * (qk_nope + v), kv_lora_rank]
        wkv_b = wkv_b.view(self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        # wkv_b shape: [n_heads, qk_nope + v_head_dim, kv_lora_rank]
        
        # Split wkv_b into k and v parts
        wkv_b_k = wkv_b[:, :self.qk_nope_head_dim, :]  # [n_heads, qk_nope, kv_lora_rank]
        wkv_b_v = wkv_b[:, -self.v_head_dim:, :]       # [n_heads, v_head_dim, kv_lora_rank]
        
        # Absorb wkv_b_k into q_nope using batched matmul (more efficient than einsum)
        # q_nope: [batch, seq, n_heads, qk_nope_head_dim]
        # wkv_b_k: [n_heads, qk_nope_head_dim, kv_lora_rank]
        # Result: [batch, seq, n_heads, kv_lora_rank]
        q_nope_t = q_nope.permute(0, 2, 1, 3)  # [batch, n_heads, seq, qk_nope]
        q_nope_absorbed = torch.matmul(q_nope_t, wkv_b_k)  # [batch, n_heads, seq, kv_lora_rank]
        
        # Update cache
        kv_cache, pe_cache = self.update_cache(kv, k_pe, start_pos)
        
        # ----- Compute Attention Scores -----
        # Use batched matmul instead of einsum for efficiency
        # q_nope_absorbed: [batch, n_heads, seq, kv_lora_rank]
        # kv_cache: [batch, total_len, kv_lora_rank]
        # scores_nope = q_nope_absorbed @ kv_cache.T
        kv_cache_t = kv_cache.unsqueeze(1).transpose(-2, -1)  # [batch, 1, kv_lora_rank, total_len]
        scores_nope = torch.matmul(q_nope_absorbed, kv_cache_t)  # [batch, n_heads, seq, total_len]
        
        # q_pe: [batch, seq, n_heads, rope_dim]
        # pe_cache: [batch, total_len, rope_dim]
        q_pe_t = q_pe.permute(0, 2, 1, 3)  # [batch, n_heads, seq, rope_dim]
        pe_cache_t = pe_cache.unsqueeze(1).transpose(-2, -1)  # [batch, 1, rope_dim, total_len]
        scores_pe = torch.matmul(q_pe_t, pe_cache_t)  # [batch, n_heads, seq, total_len]
        
        scores = (scores_nope + scores_pe) * self.softmax_scale
        
        # Apply mask if provided
        # mask shape: [seq, total_len], scores shape: [batch, n_heads, seq, total_len]
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, total_len]
        
        # Softmax
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # ----- Compute Output -----
        # Weighted sum over kv_cache using batched matmul
        # scores: [batch, n_heads, seq, total_len]
        # kv_cache: [batch, total_len, kv_lora_rank]
        kv_cache_exp = kv_cache.unsqueeze(1)  # [batch, 1, total_len, kv_lora_rank]
        output = torch.matmul(scores, kv_cache_exp)  # [batch, n_heads, seq, kv_lora_rank]
        
        # Apply wkv_b_v for value projection
        # output: [batch, n_heads, seq, kv_lora_rank]
        # wkv_b_v: [n_heads, v_head_dim, kv_lora_rank]
        # Need: output @ wkv_b_v.T -> [batch, n_heads, seq, v_head_dim]
        output = torch.matmul(output, wkv_b_v.transpose(-2, -1))  # [batch, n_heads, seq, v_head_dim]
        
        # Permute back to [batch, seq, n_heads, v_head_dim]
        output = output.permute(0, 2, 1, 3)
        
        # Flatten heads and apply output projection
        output = self.wo(output.flatten(2))
        
        return output


class MLAWithNaiveCache(nn.Module):
    """
    MLA with naive caching (stores full K and V).
    Used for correctness comparison.
    """
    
    def __init__(self, args: MLAConfig):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_local_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        
        # Same projections as MLA
        if self.q_lora_rank == 0:
            self.wq = Linear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            self.q_norm = RMSNorm(self.q_lora_rank)
            self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wkv_b = Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wo = Linear(self.n_heads * self.v_head_dim, self.dim)
        
        self.softmax_scale = self.qk_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        
        # Naive cache (full K and V)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
    
    def init_cache(self, batch_size: int, max_seq_len: int, device: str, dtype: torch.dtype):
        """Initialize K and V caches."""
        self.k_cache = torch.zeros(
            batch_size, max_seq_len, self.n_local_heads, self.qk_head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            batch_size, max_seq_len, self.n_local_heads, self.v_head_dim,
            device=device, dtype=dtype
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int, 
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with naive caching."""
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        q = torch.cat([q_nope, q_pe], dim=-1)
        
        # KV projection
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        # Decompress KV
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # Full K with expanded k_pe
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        
        # Update cache
        self.k_cache[:bsz, start_pos:end_pos] = k
        self.v_cache[:bsz, start_pos:end_pos] = v
        
        # Standard attention using matmul (more efficient than einsum)
        # q: [batch, seq, n_heads, qk_head_dim]
        # k_cache: [batch, total_len, n_heads, qk_head_dim]
        q_t = q.permute(0, 2, 1, 3)  # [batch, n_heads, seq, qk_head_dim]
        k_t = self.k_cache[:bsz, :end_pos].permute(0, 2, 3, 1)  # [batch, n_heads, qk_head_dim, total_len]
        scores = torch.matmul(q_t, k_t) * self.softmax_scale  # [batch, n_heads, seq, total_len]
        
        # mask shape: [seq, total_len], scores shape: [batch, n_heads, seq, total_len]
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, total_len]
        
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        
        # output = scores @ v_cache
        v_t = self.v_cache[:bsz, :end_pos].permute(0, 2, 1, 3)  # [batch, n_heads, total_len, v_head_dim]
        x = torch.matmul(scores, v_t)  # [batch, n_heads, seq, v_head_dim]
        x = x.permute(0, 2, 1, 3)  # [batch, seq, n_heads, v_head_dim]
        x = self.wo(x.flatten(2))
        
        return x


def create_mla_from_config(
    config: MLAConfig, 
    device: str = "cuda", 
    dtype: torch.dtype = torch.bfloat16,
    use_absorb: bool = True
) -> nn.Module:
    """
    Create an MLA module from configuration.
    
    Args:
        config: MLA configuration
        device: Target device
        dtype: Data type
        use_absorb: If True, use absorb mode (kv_cache + pe_cache)
                   If False, use naive mode (full K, V cache)
    
    Returns:
        MLA module
    """
    if use_absorb:
        model = MLA(config)
    else:
        model = MLAWithNaiveCache(config)
    
    return model.to(device=device, dtype=dtype)


def copy_weights(src_model: nn.Module, dst_model: nn.Module):
    """Copy weights from source model to destination model."""
    src_dict = src_model.state_dict()
    dst_dict = dst_model.state_dict()
    
    for key in dst_dict:
        if key in src_dict:
            dst_dict[key] = src_dict[key].clone()
    
    dst_model.load_state_dict(dst_dict)
