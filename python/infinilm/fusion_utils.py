"""
InfiniLM 专用融合工具

提供将 InfiniCore FusionScheduler 集成到 InfiniLM 模型的工具函数和上下文管理器。
"""

from typing import Optional, Dict, Any, List
import contextlib

from infinicore.fusion.fusion_scheduler import FusionScheduler
from infinicore.fusion.fusion_config import FusionConfig
from infinicore.fusion.subgraph import SubGraph
from infinicore.fusion.patterns.llm_patterns import (
    create_swiglu_pattern,
    create_add_rms_norm_pattern
)

# Re-export for use by modeling_llama.py
__all__ = [
    "FusionScheduler",
    "FusionConfig", 
    "SubGraph",
    "create_swiglu_pattern",
    "create_add_rms_norm_pattern",
    "LLMFusionContext",
    "FusionManager",
    "get_default_llm_patterns",
]

class LLMFusionContext:
    """
    LLM 推理融合上下文管理器
    
    用于在推理过程中启用或禁用算子融合。
    
    Example:
        >>> scheduler = FusionScheduler()
        >>> with LLMFusionContext(scheduler, enable=True):
        ...     # 执行模型推理，此时会自动匹配并应用融合模式
        ...     model.forward(...)
    """
    
    def __init__(self, scheduler: FusionScheduler, enable: bool = True):
        self.scheduler = scheduler
        self.enable = enable
        self._prev_state = scheduler.config.enable_fusion
        
    def __enter__(self):
        self.scheduler.config.enable_fusion = self.enable
        return self.scheduler
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scheduler.config.enable_fusion = self._prev_state

def get_default_llm_patterns() -> Dict[str, SubGraph]:
    """获取 LLM 常用的融合模式字典"""
    return {
        "swiglu": create_swiglu_pattern(),
        "add_rms_norm": create_add_rms_norm_pattern(),
    }

class FusionManager:
    """
    管理 InfiniLM 中的融合逻辑
    
    负责调度器的初始化、模式匹配和结果分发。
    """
    
    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.scheduler = FusionScheduler(self.config)
        self.patterns = get_default_llm_patterns()
        
    def run_fused(self, pattern_name: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行指定的融合模式
        
        Args:
            pattern_name: 模式名称（如 "swiglu"）
            inputs: 输入张量字典
            
        Returns:
            outputs: 输出张量字典
        """
        pattern = self.patterns.get(pattern_name)
        if pattern is None:
            raise ValueError(f"Unknown fusion pattern: {pattern_name}")
            
        return self.scheduler.dispatch(pattern, inputs)
    
    def clear_cache(self):
        """清空内核缓存"""
        self.scheduler.clear_cache()
