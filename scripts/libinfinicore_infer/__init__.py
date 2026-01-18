from .base import DataType, DeviceType, KVCacheCStruct
from .jiuge import JiugeModel, JiugeMetaCStruct, JiugeWeightsCStruct
# 为了区分, 给 ModelWeightsCStruct 别名
from .jiuge_awq import JiugeAWQModel, JiugeAWQMetaCStruct, ModelWeightsCStruct as AWQModelWeightsCStruct
# 添加 GPTQ 模块
from .jiuge_gptq import JiugeGPTQModel, JiugeGPTQMetaCStruct, ModelWeightsCStruct as GPTQModelWeightsCStruct
from .deepseek_v3 import (
    DeepSeekV3Model,
    DeepSeekV3MetaCStruct,
    DeepSeekV3WeightsCStruct,
    DeepSeekV3WeightLoaderCStruct,
    DeepSeekV3CacheCStruct,
)

__all__ = [
    "DataType",
    "DeviceType",
    "KVCacheCStruct",
    "JiugeModel",
    "JiugeMetaCStruct",
    "JiugeWeightsCStruct",
    "JiugeAWQModel",
    "JiugeAWQMetaCStruct",
    "AWQModelWeightsCStruct",
    # Add GPTQ module
    "JiugeGPTQModel",
    "JiugeGPTQMetaCStruct",
    "GPTQModelWeightsCStruct",
    "DeepSeekV3Model",
    "DeepSeekV3MetaCStruct",
    "DeepSeekV3WeightsCStruct",
    "DeepSeekV3WeightLoaderCStruct",
    "ModelRegister",
]
