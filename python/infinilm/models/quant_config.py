# coding=utf-8
# Copyright (c) 2025, InfiniCore
# BSD 3-Clause License

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type

# ---------------- 抽象层 ----------------
class QuantizationConfig(ABC):
    """InfiniCore 量化统一入口，C++ 或 Python 侧都只认这四个接口。"""
    @abstractmethod
    def get_name(self) -> str: ...
    @abstractmethod
    def get_min_capability(self) -> int: ...
    @abstractmethod
    def get_scaled_act_names(self) -> List[str]: ...
    @abstractmethod
    def get_quant_method(self) -> str:
        """返回算法名，供 C++ dispatcher 用。"""
        ...

# ---------------- 数据类 ----------------
@dataclass
class CompressedTensorsConfig(QuantizationConfig):
    """对应 HF compressed-tensors 导出格式。"""
    quant_method: str = "compressed-tensors"
    format: str = "int-quantized"
    quantization_status: str = "compressed"
    version: str = "0.11.0"
    global_compression_ratio: Optional[float] = None
    ignore: List[str] = field(default_factory=lambda: ["lm_head"])
    kv_cache_scheme: Optional[Dict[str, Any]] = None
    sparsity_config: Dict[str, Any] = field(default_factory=dict)
    transform_config: Dict[str, Any] = field(default_factory=dict)
    config_groups: Dict[str, "Group"] = field(default_factory=dict)

    @dataclass
    class TensorConfig:
        num_bits: int
        type: str
        symmetric: bool
        dynamic: bool
        strategy: str
        observer: Optional[str] = None
        observer_kwargs: Dict[str, Any] = field(default_factory=dict)
        group_size: Optional[int] = None
        block_structure: Optional[str] = None
        actorder: Optional[Any] = None

    @dataclass
    class Group:
        targets: List[str]
        weights: "CompressedTensorsConfig.TensorConfig"
        input_activations: Optional["CompressedTensorsConfig.TensorConfig"] = None
        output_activations: Optional["CompressedTensorsConfig.TensorConfig"] = None
        format: str = "int-quantized"

    @staticmethod
    def from_dict(cfg: Dict[str, Any]) -> "CompressedTensorsConfig":
        def _build_tensor(obj: Optional[Dict[str, Any]]) -> Optional["CompressedTensorsConfig.TensorConfig"]:
            return None if obj is None else CompressedTensorsConfig.TensorConfig(**obj)

        groups = {}
        for gname, gcfg in cfg.get("config_groups", {}).items():
            groups[gname] = CompressedTensorsConfig.Group(
                targets=gcfg["targets"],
                weights=_build_tensor(gcfg["weights"]),
                input_activations=_build_tensor(gcfg.get("input_activations")),
                output_activations=_build_tensor(gcfg.get("output_activations")),
                format=gcfg.get("format", "int-quantized"),
            )
        return CompressedTensorsConfig(
            quant_method=cfg["quant_method"],
            format=cfg["format"],
            quantization_status=cfg["quantization_status"],
            version=cfg["version"],
            global_compression_ratio=cfg.get("global_compression_ratio"),
            ignore=cfg.get("ignore", ["lm_head"]),
            kv_cache_scheme=cfg.get("kv_cache_scheme"),
            sparsity_config=cfg.get("sparsity_config", {}),
            transform_config=cfg.get("transform_config", {}),
            config_groups=groups,
        )

    def get_name(self) -> str:
        return self.quant_method

    def get_min_capability(self) -> int:
        return 75

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_quant_method(self) -> str:
        return self.quant_method


_QUANT_METHOD_MAP: Dict[str, Type[QuantizationConfig]] = {
    "compressed-tensors": CompressedTensorsConfig,
}

def parse_quant_config(quant_cfg: Dict[str, Any]) -> Optional[QuantizationConfig]:
    """统一解析入口，供 LlamaConfig 调用。"""
    method = quant_cfg.get("quant_method")
    cls = _QUANT_METHOD_MAP.get(method)
    if cls is None:
        return None
    
    return cls.from_dict(quant_cfg)