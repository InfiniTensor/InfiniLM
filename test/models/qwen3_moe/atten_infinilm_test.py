#!/usr/bin/env python3
# 禁用TensorFlow初始化，避免与InfiniCore冲突
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 只显示错误信息
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化

# 确保transformers库不自动导入TensorFlow
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'

import time, sys, safetensors, threading
from typing import Optional, Tuple, List
from dataclasses import dataclass
from transformers import AutoConfig
from transformers.models.qwen3_moe import modeling_qwen3_moe
from transformers import DynamicCache

# 检查Python版本，确保与InfiniCore兼容
# import sys
# print(f"Python版本: {sys.version}")
# if sys.version_info >= (3, 13):
    # print("警告: Python 3.13可能与InfiniCore不兼容，可能导致段错误")
    # # 考虑添加更严格的版本检查或退出

# ------------- InfiniCore 导入 -------------  
import infinicore as ifc
from infinicore.context import set_device, get_device
from infinicore.tensor import Tensor, from_torch
from infinicore.ops.attention import attention
from infinicore.nn import Linear
from infinicore.nn import Module as InfiniCoreModule

# 检查InfiniCore版本
print(f"InfiniCore版本: {ifc.__version__ if hasattr(ifc, '__version__') else '未知'}")

# ------------- ninetoothed 导入 ------------- 
import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed import Tensor as NTensor

WARMUPS = 10
RUNS = 100
PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [512, 0, 0, 256]}
DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],
    "pastlens": [50 for _ in range(4)]
    + [100 for _ in range(4)]
    + [200 for _ in range(4)]
    + [400 for _ in range(4)],
}

# ==================== 内存池管理 ====================
class MemoryPoolManager:
    """
    高效内存池管理器
    - 预分配内存块
    - 使用空闲列表管理
    - 支持批量分配/释放
    """
    
    def __init__(self, block_size: int, head_dim: int, num_kv_heads: int,
                 initial_slots: int, dtype: ifc.dtype, device: ifc.device):
        self.block_size = block_size
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype
        self.device = device
        
        # 内存块大小（字节）
        self.block_bytes = block_size * head_dim * num_kv_heads * dtype.itemsize
        
        # 初始容量
        self.capacity = initial_slots
        
        # 预分配内存池（使用InfiniCore API）
        self.memory = ifc.zeros(
            (self.capacity, num_kv_heads, block_size, head_dim),
            dtype=dtype, device=device
        )
        
        # 空闲槽位列表（使用InfiniCore张量实现高效管理）
        # 创建一个从0到capacity-1的序列
        self.free_slots = ifc.from_list(list(range(self.capacity)), dtype=ifc.int64, device=device)
        self.num_free = self.capacity
        
        # 统计信息
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'expansions': 0,
            'peak_usage': 0
        }
    
    def allocate(self) -> int:
        """分配一个槽位"""
        if self.num_free == 0:
            # 扩展内存池
            self._expand()
        
        slot = self.free_slots[self.num_free - 1]
        self.num_free -= 1
        
        # 更新统计信息
        self.stats['allocations'] += 1
        self.stats['peak_usage'] = max(self.stats['peak_usage'], self.capacity - self.num_free)
        
        return int(slot.item())
    
    def allocate_batch(self, count: int) -> ifc.Tensor:
        """批量分配槽位"""
        if count == 0:
            return ifc.zeros((0,), dtype=ifc.int64, device=self.device)
        
        # 确保有足够的空闲槽位
        while self.num_free < count:
            self._expand()
        
        # 分配连续的槽位
        start = self.num_free - count
        slots = self.free_slots.slice(0, start, start + count)
        self.num_free = start
        
        # 更新统计信息
        self.stats['allocations'] += count
        self.stats['peak_usage'] = max(self.stats['peak_usage'], self.capacity - self.num_free)
        
        return slots
    
    def deallocate(self, slot: int):
        """释放一个槽位"""
        if self.num_free >= self.capacity:
            return
        
        self.free_slots[self.num_free] = slot
        self.num_free += 1
        
        # 更新统计信息
        self.stats['deallocations'] += 1
    
    def deallocate_batch(self, slots: ifc.Tensor):
        """批量释放槽位"""
        if slots.numel() == 0:
            return
        
        # 确保有足够的空间
        while self.num_free + slots.numel() > self.capacity:
            self._expand()
        
        # 释放槽位
        self.free_slots.slice(0, self.num_free, self.num_free + slots.numel()).copy_(slots)
        self.num_free += slots.numel()
        
        # 更新统计信息
        self.stats['deallocations'] += slots.numel()
    
    def write_block(self, slot: int, data: ifc.Tensor):
        """写入一个块"""
        self.memory[slot] = data
    
    def write_blocks_batched(self, slots: ifc.Tensor, k_blocks: ifc.Tensor, v_blocks: ifc.Tensor):
        """批量写入块"""
        # 注意：只实现了k块的写入，v块的写入类似实现
        
        self.memory[slots] = k_blocks
    
    def read_block(self, slot: int) -> ifc.Tensor:
        """读取一个块"""
        return self.memory[slot]
    
    def gather_blocks_batched(self, slots: ifc.Tensor) -> Tuple[ifc.Tensor, ifc.Tensor]:
        """批量读取块"""
        # 注意：这里只返回了k块，v块的读取可以类似实现
        # 实际使用时需要根据需要扩展
        return self.memory[slots], self.memory[slots]
    
    def _expand(self):
        """扩展内存池"""
        # 计算新容量（1.5倍扩展）
        new_capacity = int(self.capacity * 1.5)
        
        # 预分配新内存
        new_memory = ifc.zeros(
            (new_capacity, self.num_kv_heads, self.block_size, self.head_dim),
            dtype=self.dtype, device=self.device
        )
        
        # 复制现有数据
        new_memory.slice(0, 0, self.capacity).copy_(self.memory)
        
        # 更新内存池
        self.memory = new_memory
        
        # 更新空闲槽位列表
        new_free_slots = ifc.zeros((new_capacity,), dtype=ifc.int64, device=self.device)
        # 复制现有空闲槽位
        new_free_slots.slice(0, 0, self.num_free).copy_(self.free_slots.slice(0, 0, self.num_free))
        # 填充新的空闲槽位
        new_indices = ifc.from_list(list(range(self.capacity, new_capacity)), dtype=ifc.int64, device=self.device)
        new_free_slots.slice(0, self.num_free, new_capacity).copy_(new_indices)
        self.free_slots = new_free_slots
        
        # 更新容量和空闲槽位数
        self.num_free += new_capacity - self.capacity
        self.capacity = new_capacity
        
        # 更新统计信息
        self.stats['expansions'] += 1
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'expansions': 0,
            'peak_usage': 0
        }
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'memory_pool': self.stats,
            'pool_capacity': self.capacity,
            'pool_free_slots': self.num_free
        }

# ==================== 参数调优器 参考====================
class AdaptiveParameterTuner:
    """
    自适应参数调优器
    - 根据batch size和seq len选择最优的内核参数
    - 缓存调优结果，避免重复计算
    """
    
    def __init__(self, device: ifc.device):
        self.device = device
        self._param_cache = {}  # (batch_size, seq_len, head_dim) -> (block_m, block_n, num_warps, num_stages)
    
    def get_optimal_params(self, batch_size: int, seq_len: int, head_dim: int) -> Tuple[int, int, int, int]:
        """
        获取最优参数
        参数: batch_size - 批大小, seq_len - 序列长度, head_dim - 头维度
        返回: (block_m, block_n, num_warps, num_stages)
        """
        cache_key = (batch_size, seq_len, head_dim)
        
        # 检查缓存
        if cache_key in self._param_cache:
            return self._param_cache[cache_key]
        
        # 根据经验规则选择最优参数
        # 这些规则基于Triton官方最佳实践和经验值
        if seq_len < 128:
            block_m = 64
            block_n = 32
        elif seq_len < 512:
            block_m = 128
            block_n = 64
        else:
            block_m = 256
            block_n = 64
        
        # 根据batch size调整num_warps
        if batch_size < 8:
            num_warps = 4
        elif batch_size < 32:
            num_warps = 8
        else:
            num_warps = 16
        
        # num_stages通常设置为3或4，平衡计算和内存访问
        num_stages = 3
        
        # 缓存结果
        params = (block_m, block_n, num_warps, num_stages)
        self._param_cache[cache_key] = params
        
        return params

# ==================== 内核缓存 ====================
class KernelCache:
    """
    内核缓存管理器
    - 缓存已创建的注意力内核
    - 避免重复创建和编译内核
    """
    _cache = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_or_create(cls, head_dim: int, block_m: int, block_n: int, num_warps: int, num_stages: int):
        """
        获取或创建内核
        参数: head_dim, block_m, block_n, num_warps, num_stages
        返回: 创建好的注意力内核
        """
        key = (head_dim, block_m, block_n, num_warps, num_stages)
        
        with cls._lock:
            if key not in cls._cache:
                # 创建新内核
                cls._cache[key] = create_attention_kernel_online_softmax(head_dim, block_m, block_n, num_warps, num_stages)
            
            return cls._cache[key]

# ==================== 批处理优化 ====================
@dataclass
class AttentionRequest:
    """注意力请求数据类"""
    q: ifc.Tensor  # (1, num_heads, q_len, head_dim)
    k: ifc.Tensor  # (1, num_kv_heads, k_len, head_dim)
    v: ifc.Tensor  # (1, num_kv_heads, k_len, head_dim)
    seq_len: int
    is_causal: bool
    past_key_values: Optional[DynamicCache] = None
    attention_mask: Optional[ifc.Tensor] = None

class BatchRequestMerger:
    """
    批处理请求合并器
    - 合并小请求以提高GPU利用率
    - 动态批处理大小调整
    """
    
    def __init__(self, max_batch_size: int = 64):
        self.max_batch_size = max_batch_size
        self.pending_requests: List[AttentionRequest] = []
    
    def add_request(self, request: AttentionRequest):
        """添加请求"""
        self.pending_requests.append(request)
    
    def should_flush(self) -> bool:
        """检查是否应该刷新批处理"""
        return len(self.pending_requests) >= self.max_batch_size
    
    def merge_requests(self) -> Optional[Tuple[ifc.Tensor, ifc.Tensor, 
                                                ifc.Tensor, List[AttentionRequest], bool]]:
        """
        合并待处理请求
        返回: (q_merged, k_merged, v_merged, original_requests, is_causal)
        """
        if not self.pending_requests:
            return None
        
        requests = self.pending_requests
        self.pending_requests = []
        
        # 按序列长度分组以减少padding
        requests.sort(key=lambda r: r.seq_len, reverse=True)
        
        batch_size = len(requests)
        max_q_len = max(r.q.shape[2] for r in requests)
        max_k_len = max(r.k.shape[2] for r in requests)
        
        # 获取维度信息
        sample = requests[0]
        num_heads = sample.q.shape[1]
        num_kv_heads = sample.k.shape[1]
        head_dim = sample.q.shape[3]
        device = sample.q.device
        dtype = sample.q.dtype
        
        # 预分配合并后的张量（使用InfiniCore API）
        q_merged = ifc.zeros(
            (batch_size, num_heads, max_q_len, head_dim),
            dtype=dtype, device=device
        )
        k_merged = ifc.zeros(
            (batch_size, num_kv_heads, max_k_len, head_dim),
            dtype=dtype, device=device
        )
        v_merged = ifc.zeros_like(k_merged)
        
        # 填充数据
        for i, req in enumerate(requests):
            q_len = req.q.shape[2]
            k_len = req.k.shape[2]
            # 移除batch维度并填充数据
            q_slice = q_merged.slice(0, i, i+1).slice(2, 0, q_len)
            q_slice.copy_(req.q.slice(0, 0, 1))
            
            k_slice = k_merged.slice(0, i, i+1).slice(2, 0, k_len)
            k_slice.copy_(req.k.slice(0, 0, 1))
            
            v_slice = v_merged.slice(0, i, i+1).slice(2, 0, k_len)
            v_slice.copy_(req.v.slice(0, 0, 1))
        
        is_causal = requests[0].is_causal
        
        return q_merged, k_merged, v_merged, requests, is_causal

# ==================== 辅助函数 ====================
def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--nvidia", action="store_true", help="nvidia gpu")
    p.add_argument("--tian", action="store_true", help="tian gpu")
    p.add_argument("--metax", action="store_true", help="metax gpu")
    return p.parse_args()

def device_synchronize(_device):
    if isinstance(_device, ifc.device):
        ifc.sync_device(_device)

def device_empty_cache(_device):
    if isinstance(_device, ifc.device):
        # InfiniCore设备不需要特殊处理，由内部管理
        pass

def rotate_half(x):
    # 使用InfiniCore API实现rotate_half操作
    # 检查输入是否为InfiniCore张量
    if isinstance(x, ifc.Tensor):
        # 使用切片和数学运算实现rotate_half
        # 获取维度信息
        last_dim = x.shape[-1]
        # 创建结果张量
        result = ifc.zeros_like(x)
        # 计算切片范围
        half_dim = last_dim // 2
        # 第一部分：-x2
        result.slice(-1, 0, half_dim).copy_(-x.slice(-1, half_dim, half_dim * 2))
        # 第二部分：x1
        result.slice(-1, half_dim, last_dim).copy_(x.slice(-1, 0, half_dim))
        return result
    # 否则使用ninetoothed API
    elif isinstance(x, nt.Tensor):
        # ninetoothed实现
        x1, x2 = x.split(2, dim=-1)
        return nt.concat((-x2, x1), dim=-1)
    else:
        raise TypeError(f"Unsupported tensor type: {type(x)}")

# ==================== ninetoothed kernel ====================
class NTQwen3MoeAttention(InfiniCoreModule):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        
        # 使用InfiniCore的Linear层
        self.q_proj = Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.k_proj = Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.v_proj = Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.attention_bias
        )
        self.o_proj = Linear(
            self.num_heads * self.head_dim, 
            self.hidden_size, 
            bias=config.attention_bias
        )
        
        # 初始化参数调优器
        self.param_tuner = None
        
        # 移除固定内核初始化，改为动态创建
        self._attention_kernel = None
        
        # 添加批量优化支持
        self.batch_merger = BatchRequestMerger(max_batch_size=64)
        
        # 添加内存池管理（用于KV缓存优化）
        self.memory_pool = None
        
        # InfiniCore 设备初始化
        self.infini_device = None
        self.use_infinicore = False
        

    def apply_rope(self, q, k, cos, sin):
        # 检查输入是否为InfiniCore张量
        if isinstance(q, ifc.Tensor) and isinstance(k, ifc.Tensor):
            # 使用InfiniCore API实现RoPE
            q_rot = rotate_half(q)
            k_rot = rotate_half(k)
            q_embed = (q * cos) + (q_rot * sin)
            k_embed = (k * cos) + (k_rot * sin)
            return q_embed, k_embed
        # 否则使用ninetoothed API
        elif isinstance(q, nt.Tensor) and isinstance(k, nt.Tensor):
            q_rot = rotate_half(q)
            k_rot = rotate_half(k)
            q_embed = (q * cos) + (q_rot * sin)
            k_embed = (k * cos) + (k_rot * sin)
            return q_embed, k_embed
        else:
            raise TypeError(f"Unsupported tensor type: {type(q)}, {type(k)}")

    def _init_infinicore(self, device):
        """
        初始化InfiniCore设备
        """
        if self.infini_device is None:
            # 参考InfiniLM示例，使用字符串作为设备类型
            if str(device).startswith("cuda"):
                device_str = "cuda"
            elif str(device).startswith("musa"):
                device_str = "musa"
            elif str(device).startswith("tian"):
                device_str = "tian"
            else:
                device_str = "cpu"
            
            # 创建InfiniCore设备
            self.infini_device = ifc.device(device_str, device.index if device.index is not None else 0)
            
            # 设置当前设备
            set_device(self.infini_device)
            self.use_infinicore = True
    
    def forward(
        self,
        hidden_states: ifc.Tensor,
        position_embeddings: Optional[Tuple[ifc.Tensor, ifc.Tensor]] = None,
        attention_mask: Optional[ifc.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_infinicore: bool = True,
        **kwargs,
    ):
        """
        前向传播，InfiniCore加速
        """
        if use_infinicore:
            return self.forward_infinicore(
                hidden_states, position_embeddings, attention_mask, 
                past_key_values, output_attentions, use_cache, **kwargs
            )
        else:
            return self.forward_original(
                hidden_states, position_embeddings, attention_mask, 
                past_key_values, output_attentions, use_cache, **kwargs
            )
    
    def forward_original(
        self,
        hidden_states: ifc.Tensor,
        position_embeddings: Optional[Tuple[ifc.Tensor, ifc.Tensor]] = None,
        attention_mask: Optional[ifc.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        原始的forward方法，使用ninetoothed实现
        """
        bsz, q_len, _ = hidden_states.shape
        
        # 初始化内存池（如果尚未初始化）
        if self.memory_pool is None:
            self.memory_pool = MemoryPoolManager(
                block_size=64,  # 默认块大小
                head_dim=self.head_dim,
                num_kv_heads=self.num_key_value_heads,
                initial_slots=4096,  # 初始槽位数
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
        
        # 初始化参数调优器（如果尚未初始化）
        if self.param_tuner is None:
            self.param_tuner = AdaptiveParameterTuner(device=hidden_states.device)
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 使用InfiniCore API进行形状调整
        # view操作：(bsz, q_len, hidden_size) -> (bsz, q_len, num_heads, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        # transpose操作：(bsz, q_len, num_heads, head_dim) -> (bsz, num_heads, q_len, head_dim)
        query_states = query_states.permute(0, 2, 1, 3)
        key_states = key_states.permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)
        
        if position_embeddings is not None:
            sin, cos = position_embeddings
            query_states, key_states = self.apply_rope(query_states, key_states, cos, sin)
        
        if past_key_values is not None and use_cache:
            cache_kwargs = {"sin": sin, "cos": cos} if position_embeddings is not None else {}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        num_groups = self.num_heads // self.num_key_value_heads
        if num_groups > 1:
            # 使用InfiniCore API进行repeat_interleave操作
            key_states = key_states.repeat_interleave(num_groups, dim=1)
            value_states = value_states.repeat_interleave(num_groups, dim=1)
        
        is_causal = False
        # 使用InfiniCore API创建空张量
        output = ifc.zeros_like(query_states)
        
        # 获取k_len（当前序列长度 + past序列长度）
        k_len = key_states.shape[2]
        
        # 获取最优参数
        block_m, block_n, num_warps, num_stages = self.param_tuner.get_optimal_params(
            batch_size=bsz,
            seq_len=max(q_len, k_len),
            head_dim=self.head_dim
        )
        
        # 从内核缓存获取或创建内核
        self._attention_kernel = KernelCache.get_or_create(
            head_dim=self.head_dim,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        # 调用注意力内核
        self._attention_kernel(query_states, key_states, value_states, is_causal, output)
        
        # 使用InfiniCore API进行形状调整
        attn_output = output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_values if use_cache else None
    
    def forward_infinicore(
        self,
        hidden_states: ifc.Tensor,
        position_embeddings: Optional[Tuple[ifc.Tensor, ifc.Tensor]] = None,
        attention_mask: Optional[ifc.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        """
        使用InfiniCore加速的forward方法
        """
        bsz, q_len, _ = hidden_states.shape
        
        # 初始化InfiniCore设备
        self._init_infinicore(hidden_states.device)
        
        # 前向传播获取query, key, value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 调整形状
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # RoPE已经应用到hidden_states上，不需要再次应用
        pass
        
        # 处理多头注意力 - 注意：InfiniCore attention算子期望k和v使用key_value_heads，而不是num_heads
        # 需要将key_states和value_states的头数扩展到num_heads
        # 多头注意力的处理由InfiniCore attention算子内部完成
        num_groups = self.num_heads // self.num_key_value_heads
        # 保存num_groups用于后续处理
        
        # 获取past_key_values的长度
        past_len = 0
        if past_key_values is not None:
            # 获取past_key_values的长度
            past_len = past_key_values.get_seq_length()
        
        # 调整张量形状以匹配InfiniCore attention算子的预期格式
        # InfiniCore attention算子期望的形状是：
        # q: [n_q_head, seq_len, head_dim]
        # k: [n_kv_head, seq_len, head_dim]
        # v: [n_kv_head, seq_len, head_dim]
        # k_cache: [n_kv_head, cache_size, head_dim]
        # v_cache: [n_kv_head, cache_size, head_dim]
        
        # 对于batch_size > 1的情况，我们需要逐个处理每个样本
        # batch_size = 1
        if bsz != 1:
            raise NotImplementedError("InfiniCore attention operator currently only supports batch_size = 1")
        
        # 获取key_value_heads和num_heads
        n_q_head = self.num_heads
        n_kv_head = self.num_key_value_heads
        
        # 移除batch维度（batch_size = 1）
        # 使用InfiniCore API进行squeeze操作
        query_states = query_states.squeeze(0)  # [num_heads, q_len, head_dim]
        key_states = key_states.squeeze(0)      # [num_key_value_heads, q_len, head_dim] - 保持key_value_heads
        value_states = value_states.squeeze(0)  # [num_key_value_heads, q_len, head_dim] - 保持key_value_heads
        
        # 确保所有张量的形状完全符合InfiniCore attention算子的预期
        # query_states应该是 [n_q_head, q_len, head_dim]
        # key_states和value_states应该是 [n_kv_head, q_len, head_dim]
        assert query_states.shape == (n_q_head, q_len, self.head_dim), f"query_states shape mismatch: {query_states.shape} != ({n_q_head}, {q_len}, {self.head_dim})"
        assert key_states.shape == (n_kv_head, q_len, self.head_dim), f"key_states shape mismatch: {key_states.shape} != ({n_kv_head}, {q_len}, {self.head_dim})"
        assert value_states.shape == (n_kv_head, q_len, self.head_dim), f"value_states shape mismatch: {value_states.shape} != ({n_kv_head}, {q_len}, {self.head_dim})"
        
        # 已经是InfiniCore张量，不需要转换
        q_if = query_states
        k_if = key_states
        v_if = value_states
        
        # 创建k_cache和v_cache张量（根据测试文件，这些是必需的）
        # 对于prefill阶段，pos=0；对于decode阶段，pos=past_len
        pos = past_len
        
        # 计算缓存大小
        cache_size = past_len + q_len
        
        # 获取InfiniCore设备
        device = q_if.device
        
        # 创建足够大的缓存张量（使用InfiniCore API）
        # 注意：k_cache和v_cache应该使用key-value头数，而不是查询头数
        k_cache_shape = (self.num_key_value_heads, cache_size, self.head_dim)
        k_cache_if = ifc.zeros(k_cache_shape, dtype=q_if.dtype, device=device)
        
        v_cache_shape = (self.num_key_value_heads, cache_size, self.head_dim)
        v_cache_if = ifc.zeros(v_cache_shape, dtype=q_if.dtype, device=device)
        
        # 确保所有张量都在同一设备上
        assert k_if.device == device, f"k_if device mismatch: {k_if.device} != {device}"
        assert v_if.device == device, f"v_if device mismatch: {v_if.device} != {device}"
        assert k_cache_if.device == device, f"k_cache_if device mismatch: {k_cache_if.device} != {device}"
        assert v_cache_if.device == device, f"v_cache_if device mismatch: {v_cache_if.device} != {device}"
        
        # 确保pos参数是整数
        assert isinstance(pos, int), f"pos must be int, got {type(pos)}"
        
        # 调用attention算子，不使用out参数，让函数返回结果
        # 这样可以避免一些内存管理问题
        output_result = attention(q_if, k_if, v_if, k_cache_if, v_cache_if, pos)
        
        # 预分配输出张量，形状与返回结果一致
        output_if = ifc.zeros_like(output_result)
        
        # 将结果复制到output_if中
        output_if.copy_(output_result)
        
        # 调整output形状：[seq_len, n_q_head, head_dim] -> [n_q_head, seq_len, head_dim]
        output_if = output_if.permute(1, 0, 2)
        
        # 添加batch维度：[n_q_head, seq_len, head_dim] -> [1, n_q_head, seq_len, head_dim]
        output_if = output_if.unsqueeze(0)
        
        output = output_if
        
        # 更新KV缓存
        if past_key_values is not None and use_cache:
            # 恢复key_states和value_states的batch维度，然后更新缓存
            key_states = key_states.unsqueeze(0)
            value_states = value_states.unsqueeze(0)
            # 不再需要cache_kwargs，因为RoPE已经应用到hidden_states上
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        
        # 后续处理
        attn_output = output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_values if use_cache else None
        
    def forward_batched(
        self,
        requests: List[AttentionRequest],
        use_infinicore: bool = True
    ):
        """
        批量处理多个注意力请求，InfiniCore加速
        参数: requests - AttentionRequest列表
        返回: 处理结果列表
        """
        # 添加请求到合并器
        for req in requests:
            self.batch_merger.add_request(req)
        
        # 合并并处理请求
        merged_data = self.batch_merger.merge_requests()
        if not merged_data:
            return []
        
        q_merged, k_merged, v_merged, original_requests, is_causal = merged_data
        
        if use_infinicore:
            return self._forward_batched_infinicore(
                q_merged, k_merged, v_merged, original_requests, is_causal
            )
        else:
            return self._forward_batched_original(
                q_merged, k_merged, v_merged, original_requests, is_causal
            )
    
    def _forward_batched_original(
        self,
        q_merged: ifc.Tensor,
        k_merged: ifc.Tensor,
        v_merged: ifc.Tensor,
        original_requests: List[AttentionRequest],
        is_causal: bool
    ):
        """
        原始的批量处理方法，使用ninetoothed实现
        """
        # 初始化内存池（如果尚未初始化）
        if self.memory_pool is None:
            self.memory_pool = MemoryPoolManager(
                block_size=64,  # 默认块大小
                head_dim=self.head_dim,
                num_kv_heads=self.num_key_value_heads,
                initial_slots=4096,  # 初始槽位数
                dtype=q_merged.dtype,
                device=q_merged.device
            )
        
        # 初始化参数调优器（如果尚未初始化）
        if self.param_tuner is None:
            self.param_tuner = AdaptiveParameterTuner(device=q_merged.device)
        
        # InfiniCore attention算子期望k和v使用num_kv_heads，而不是num_heads
        # 不需要将k_merged和v_merged的头数扩展到num_heads
        num_heads = q_merged.shape[1]
        num_kv_heads = k_merged.shape[1]
        num_groups = num_heads // num_kv_heads
        # 保存num_groups用于后续处理
        
        # 预分配输出张量
        output_merged = ifc.empty_like(q_merged, dtype=v_merged.dtype)
        
        # 获取k_len和q_len
        batch_size = q_merged.shape[0]
        q_len = q_merged.shape[2]
        k_len = k_merged.shape[2]
        
        # 获取最优参数
        block_m, block_n, num_warps, num_stages = self.param_tuner.get_optimal_params(
            batch_size=batch_size,
            seq_len=max(q_len, k_len),
            head_dim=self.head_dim
        )
        
        # 从内核缓存获取或创建内核
        attention_kernel = KernelCache.get_or_create(
            head_dim=self.head_dim,
            block_m=block_m,
            block_n=block_n,
            num_warps=num_warps,
            num_stages=num_stages
        )
        
        # 调用注意力内核处理整个批次
        attention_kernel(q_merged, k_merged, v_merged, is_causal, output_merged)
        
        # 将结果拆分回原始请求
        results = []
        for i, req in enumerate(original_requests):
            # 提取对应请求的结果（移除padding）
            q_len = req.q.shape[2]
            batch_result = output_merged[i, :, :q_len, :]
            
            # 转换回原始格式
            batch_result = batch_result.transpose(0, 1).contiguous()
            batch_result = batch_result.reshape(1, q_len, self.num_heads * self.head_dim)
            batch_result = self.o_proj(batch_result)
            
            results.append(batch_result)
        
        return results
    
    def _forward_batched_infinicore(
        self,
        q_merged: ifc.Tensor,
        k_merged: ifc.Tensor,
        v_merged: ifc.Tensor,
        original_requests: List[AttentionRequest],
        is_causal: bool
    ):
        """
        使用InfiniCore加速的批量处理方法
        """
        # 初始化InfiniCore设备
        self._init_infinicore(q_merged.device)
        
        # 检查是否需要重复key和value（如果num_heads > num_key_value_heads）
        num_heads = q_merged.shape[1]
        num_kv_heads = k_merged.shape[1]
        num_groups = num_heads // num_kv_heads
        
        if num_groups > 1:
            k_merged = k_merged.repeat_interleave(num_groups, dim=1)
            v_merged = v_merged.repeat_interleave(num_groups, dim=1)
        
        # 获取批次大小
        batch_size = q_merged.shape[0]
        
        # 逐个处理每个样本，因为InfiniCore attention算子目前只支持batch_size=1
        results = []
        for i in range(batch_size):
            # 提取单个样本
            q_single = q_merged[i:i+1]  # [1, num_heads, q_len, head_dim]
            k_single = k_merged[i:i+1]  # [1, num_heads, q_len, head_dim]
            v_single = v_merged[i:i+1]  # [1, num_heads, q_len, head_dim]
            
            # 获取当前样本的序列长度
            req = original_requests[i]
            q_len = req.q.shape[2]
            
            # 调整张量形状以匹配InfiniCore attention算子的预期格式
            # 移除batch维度
            q_single = q_single.squeeze(0)  # [num_heads, q_len, head_dim]
            k_single = k_single.squeeze(0)  # [num_heads, q_len, head_dim]
            v_single = v_single.squeeze(0)  # [num_heads, q_len, head_dim]
            
            # 已经是InfiniCore张量，不需要转换
            q_if = q_single
            k_if = k_single
            v_if = v_single
            
            # 获取past_len
            past_len = 0
            if req.past_key_values:
                past_len = req.past_key_values.get_seq_length()
            pos = past_len
            
            # 计算缓存大小
            cache_size = past_len + q_len
            
            # 获取InfiniCore设备
            device = q_if.device
            
            # 确保所有张量都在同一设备上
            assert k_if.device == device, f"k_if device mismatch: {k_if.device} != {device}"
            assert v_if.device == device, f"v_if device mismatch: {v_if.device} != {device}"
            
            # 确保pos参数是整数
            assert isinstance(pos, int), f"pos must be int, got {type(pos)}"
            
            # 创建k_cache和v_cache张量（使用InfiniCore API）
            # k_cache和v_cache使用key-value头数，而不是查询头数
            k_cache_shape = (num_kv_heads, cache_size, self.head_dim)
            k_cache_if = ifc.zeros(k_cache_shape, dtype=q_if.dtype, device=device)
            
            v_cache_shape = (num_kv_heads, cache_size, self.head_dim)
            v_cache_if = ifc.zeros(v_cache_shape, dtype=q_if.dtype, device=device)
            
            # 调用InfiniCore的attention算子，不使用out参数
            output_result = attention(q_if, k_if, v_if, k_cache_if, v_cache_if, pos)
            
            # 预分配输出张量，形状与返回结果一致
            output_if = ifc.zeros_like(output_result)
            
            # 将结果复制到output_if中
            output_if.copy_(output_result)
            
            # 调整output形状：[seq_len, n_q_head, head_dim] -> [n_q_head, seq_len, head_dim]
            output_if = output_if.permute(1, 0, 2)
            
            # 添加batch维度：[n_q_head, seq_len, head_dim] -> [1, n_q_head, seq_len, head_dim]
            output_if = output_if.unsqueeze(0)
            
            # 已经是InfiniCore张量，不需要转换
            output = output_if
            
            # 后续处理
            batch_result = output.permute(1, 2, 0, 3).contiguous()
            batch_result = batch_result.view(1, q_len, self.num_heads * self.head_dim)
            batch_result = self.o_proj(batch_result)
            
            results.append(batch_result)
        
        return results

def create_attention_kernel_online_softmax(head_dim=128, block_m=128, block_n=64, num_warps=4, num_stages=3):
    # 使用传入的参数或默认值
    BLOCK_SIZE_M = block_m
    BLOCK_SIZE_N = block_n

    q_ = NTensor(4, shape_options=(
        None,
        None,
        {"constexpr": True},
        {"constexpr": True, "upper_bound": head_dim},
    ))
    k_ = NTensor(4, shape_options=(
        None,
        None,
        {"constexpr": True},
        {"constexpr": True, "upper_bound": head_dim},
    ))
    v_ = NTensor(4, shape_options=(
        None,
        None,
        {"constexpr": True},
        {"constexpr": True, "upper_bound": head_dim},
    ))
    o_ = NTensor(4, shape_options=(
        None,
        None,
        {"constexpr": True},
        {"constexpr": True, "upper_bound": head_dim},
    ))
    is_causal_ = NTensor(0, constexpr=True)

    def arrangement(q, k, v, is_causal, o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N):
        def arrange_q_or_o(input_tensor):
            arranged = input_tensor.tile((1, 1, BLOCK_SIZE_M, -1))
            arranged.dtype = arranged.dtype.squeeze((0, 1))
            return arranged

        def arrange_k_or_v(input_tensor):
            arranged = (
                input_tensor.tile((1, 1, BLOCK_SIZE_N, -1))
                .tile((1, 1, -1, -1))
                .expand((-1, -1, q_arranged.shape[-2], -1))
            )
            arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
            arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
            return arranged

        q_arranged = arrange_q_or_o(q)
        return (
            q_arranged,
            arrange_k_or_v(k),
            arrange_k_or_v(v),
            is_causal,
            arrange_q_or_o(o),
        )

    def application(q, k, v, is_causal, o):
        q_loaded = (q * 1.44269504089).to(q.dtype)

        acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
        l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
        m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

        # 动态块数
        num_blocks = k.shape[0]
        
        for bi in range(num_blocks):
            qk = ntl.dot(q_loaded, ntl.trans(k[bi]))
            qk = ntl.where(k[bi].offsets(-2) < k.source.shape[-2], qk, float("-inf"))

            if is_causal:
                mask = q.offsets(-2)[:, None] >= k[bi].offsets(-2)[None, :]
                qk = ntl.where(mask, qk, float("-inf"))

            m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
            p = ntl.exp2(qk - m_ij[:, None])
            l_ij = ntl.sum(p, 1)

            alpha = ntl.exp2(m_i - m_ij)
            acc = acc * alpha[:, None] + ntl.dot(p.to(v[bi].dtype), v[bi])
            m_i = m_ij
            l_i = l_i * alpha + l_ij

        acc /= l_i[:, None]
        o = acc

    tensors = (q_, k_, v_, is_causal_, o_)
    return nt.make(arrangement, application, tensors, num_warps=num_warps, num_stages=num_stages)

# ==================== 构造模型 ====================
def create_nt_attention(dir_path, *, device, dtype=ifc.bfloat16):
    config = AutoConfig.from_pretrained(dir_path)
    config.num_hidden_layers = 1
    
    model = NTQwen3MoeAttention(config, layer_idx=0)
    model.eval()
    
    tensors = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"):
            continue
        fpath = os.path.join(dir_path, fname)
        with safetensors.safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                if "model.layers.0.self_attn." in key:
                    # Convert torch tensor to infinicore tensor
                    torch_tensor = f.get_tensor(key)
                    infini_tensor = from_torch(torch_tensor)
                    tensors[key[len("model.layers.0.self_attn.") :]] = infini_tensor
        break
    model.load_state_dict(tensors, strict=False)
    
    # 使用 InfiniCore 的 RoPE 实现
    from infinicore.nn.modules.rope import RoPE
    rotary_emb = RoPE(
        max_position_embeddings=4096,  # 使用一个合理的默认值
        rope_theta=config.rope_theta,  # 从模型配置获取
        head_dim=config.head_dim,      # 从模型配置获取
        device=device,
        dtype=dtype
    )
    
    return model, rotary_emb

# ==================== 生成输入 ====================
def generate_attention_input_nt(model, rotary_emb, testcase, device, dtype=ifc.bfloat16):
    config = model.config
    hidden_size = config.hidden_size
    head_dim = config.head_dim
    num_key_value_heads = config.num_key_value_heads
    bs = 1
    
    req_list = []
    for seq_lens, past_lens in zip(testcase["seqlens"], testcase["pastlens"]):
        hidden_states = ifc.rand(
            (bs, seq_lens, hidden_size), dtype=dtype, device=device
        )
        
        past_key_values = DynamicCache(config=config)
        if past_lens > 0:
            key_states = ifc.rand(
                (bs, num_key_value_heads, past_lens, head_dim), dtype=dtype, device=device
            )
            value_states = ifc.rand(
                (bs, num_key_value_heads, past_lens, head_dim), dtype=dtype, device=device
            )
            past_key_values.update(key_states, value_states, 0)
        
        req = {
            "hidden_states": hidden_states,
            "attention_mask": None,
            "past_key_values": past_key_values,
        }
        req_list.append(req)
    
    return req_list
    
def generate_batched_attention_input_nt(model, rotary_emb, testcase, device, dtype=ifc.bfloat16):
    """
    生成批量注意力请求
    返回: AttentionRequest对象列表
    """
    config = model.config
    hidden_size = config.hidden_size
    head_dim = config.head_dim
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    bs = 1
    
    attention_reqs = []
    for seq_lens, past_lens in zip(testcase["seqlens"], testcase["pastlens"]):
        # 生成隐藏状态（使用InfiniCore API）
        hidden_states = ifc.rand(
            (bs, seq_lens, hidden_size), dtype=dtype, device=device
        )
        
        # 生成past_key_values
        past_key_values = DynamicCache(config=config)
        if past_lens > 0:
            # 生成key_states和value_states（使用InfiniCore API）
            key_states = ifc.rand(
                (bs, num_key_value_heads, past_lens, head_dim), dtype=dtype, device=device
            )
            value_states = ifc.rand(
                (bs, num_key_value_heads, past_lens, head_dim), dtype=dtype, device=device
            )
            past_key_values.update(key_states, value_states, 0)
        
        # 生成position_ids和rotary embeddings
        cache_lens = past_key_values.get_seq_length()
        # 使用Python列表生成position_ids，然后转换为InfiniCore张量
        position_ids = ifc.from_list(
            list(range(cache_lens, cache_lens + seq_lens)), dtype=ifc.int64, device=device
        ).reshape((bs, seq_lens))
        # 使用InfiniCore的RoPE实现，直接修改hidden_states
        rotary_emb(hidden_states, position_ids)
        # RoPE已经应用到hidden_states上，不需要position_embeddings
        position_embeddings = None
        
        # 前向传播得到query, key, value
        query_states = model.q_proj(hidden_states)
        key_states = model.k_proj(hidden_states)
        value_states = model.v_proj(hidden_states)
        
        query_states = query_states.view(bs, seq_lens, num_heads, head_dim).permute(0, 2, 1, 3)
        key_states = key_states.view(bs, seq_lens, num_key_value_heads, head_dim).permute(0, 2, 1, 3)
        value_states = value_states.view(bs, seq_lens, num_key_value_heads, head_dim).permute(0, 2, 1, 3)
        
        # 应用rope
        query_states, key_states = model.apply_rope(query_states, key_states, cos_table, sin_table)
        
        # 更新past_key_values
        cache_kwargs = {"sin": cos_table, "cos": sin_table} if position_embeddings is not None else {}
        key_states, value_states = past_key_values.update(key_states, value_states, model.layer_idx, cache_kwargs)
        
        # 创建AttentionRequest对象
        attention_req = AttentionRequest(
            q=query_states,
            k=key_states,
            v=value_states,
            seq_len=seq_lens,
            is_causal=False,
            past_key_values=past_key_values,
            attention_mask=None
        )
        attention_reqs.append(attention_req)
    
    return attention_reqs


# ==================== 基准测试代码 ====================
def benchmark_Qwen3attention_optimized(
    model, rotary_emb, test_cases, device, dtype=ifc.bfloat16, mode="prefill"
):
    """
    整合测试
    整合了：批量处理、内存池、自适应参数调优、内核缓存和InfiniCore加速
    参数:
        - model: 注意力模型
        - rotary_emb: 旋转位置编码
        - test_cases: 测试用例
        - device: 设备
        - dtype: 数据类型
        - mode: "prefill" 或 "decode"
    """
    if mode == "prefill":
        # 生成批量注意力请求
        attention_reqs = generate_batched_attention_input_nt(model, rotary_emb, test_cases, device, dtype)
        
        # 预热
        for _ in range(WARMUPS):
            # 重置past_key_values
            for i, req in enumerate(attention_reqs):
                origin_len = test_cases["pastlens"][i]
                req.past_key_values.crop(origin_len)
            
            # 使用批量处理，指定使用infiniCore
            results = model.forward_batched(attention_reqs, use_infinicore=True)
            
            # 将结果转移到CPU
            for result in results:
                result_host = result.to("cpu")

        ifc.synchronize(device)

        # 正式测试
        time_consuming = 0
        for _ in range(RUNS):
            # 重置past_key_values
            for i, req in enumerate(attention_reqs):
                origin_len = test_cases["pastlens"][i]
                req.past_key_values.crop(origin_len)
            
            ifc.synchronize(device)
            start_time = time.time()
            
            # 使用批量处理，指定使用infiniCore
            results = model.forward_batched(attention_reqs, use_infinicore=True)
            
            ifc.synchronize(device)
            end_time = time.time()
            time_consuming += end_time - start_time

        out_token_count = RUNS * len(attention_reqs)
        latency = time_consuming * 1000 / out_token_count

        print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, Optimized InfiniCore-Attention (Batched), average TTFT: {round(latency, 2)} ms\n")
    
    elif mode == "decode":
        # 生成批量注意力请求
        attention_reqs = generate_batched_attention_input_nt(model, rotary_emb, test_cases, device, dtype)
        
        # 预热
        for _ in range(WARMUPS):
            results = model.forward_batched(attention_reqs, use_infinicore=True)
            
            # 将结果转移到CPU
            for result in results:
                result_host = result.to("cpu")

        # 重置past_key_values
        for i, req in enumerate(attention_reqs):
            origin_len = test_cases["pastlens"][i]
            req.past_key_values.crop(origin_len)

        ifc.synchronize(device)
        start_time = time.time()

        # 正式测试
        for _ in range(RUNS):
            results = model.forward_batched(attention_reqs, use_infinicore=True)
            
            # 注意：在decode benchmark中，我们不需要更新请求的query states用于下一轮
            # 因为每个迭代都是独立的测试，我们只需要计算吞吐量
            pass

        ifc.synchronize(device) 
        end_time = time.time()

        time_consuming = end_time - start_time
        out_token_count = RUNS * len(attention_reqs)
        throughput = out_token_count / time_consuming

        print(f"\t WARMUPS={WARMUPS} RUNS={RUNS}, Optimized InfiniCore-Attention (Batched), average throughput: {round(throughput, 2)} tok/s \n")
    
    return []

# ==================== 主入口 ====================
if __name__ == "__main__":
    args = get_args()
    print(args)

    model_path = args.model_path
    dtype = ifc.bfloat16
    
    if args.nvidia:
        device = "cuda:0"
    elif args.tian:
        device = "tian:0"
    elif args.metax:
        device = "musa:0"
    else:
        print("Usage:  python attention_test_ninetoothed.py [--nvidia | --tian | --metax] --model_path <path/to/model_path>")
        sys.exit(1)
    
    device = ifc.device(device)
    
    model, rotary_emb = create_nt_attention(model_path, device=device, dtype=dtype)
    
    print("\n")
    print("*" * 130)
    print("Test Optimized InfiniCore-Attention ")
    print("*" * 130)
    print(f"Test Case PREFILL_TESTCASES : {PREFILL_TESTCASES}")
    benchmark_Qwen3attention_optimized(model, rotary_emb, PREFILL_TESTCASES, device, dtype=dtype, mode="prefill")
    
    print("\n")
    print("=" * 130)
    print(f"Test DECODE_TESTCASES: {DECODE_TESTCASES}")
    benchmark_Qwen3attention_optimized(model, rotary_emb, DECODE_TESTCASES, device, dtype=dtype, mode="decode")

    del model
    ifc.empty_cache(device)
