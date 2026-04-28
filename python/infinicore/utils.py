import numpy as np
import torch

import infinicore

try:
    import ml_dtypes
except ModuleNotFoundError:
    ml_dtypes = None


def to_torch_dtype(infini_dtype):
    """Convert infinicore data type to PyTorch data type"""
    if infini_dtype == infinicore.float16:
        return torch.float16
    elif infini_dtype == infinicore.float32:
        return torch.float32
    elif infini_dtype == infinicore.bfloat16:
        return torch.bfloat16
    elif infini_dtype == infinicore.int8:
        return torch.int8
    elif infini_dtype == infinicore.int16:
        return torch.int16
    elif infini_dtype == infinicore.int32:
        return torch.int32
    elif infini_dtype == infinicore.int64:
        return torch.int64
    elif infini_dtype == infinicore.uint8:
        return torch.uint8
    else:
        raise ValueError(f"Unsupported infinicore dtype: {infini_dtype}")


def to_infinicore_dtype(torch_dtype):
    """Convert PyTorch data type to infinicore data type"""
    if torch_dtype == torch.float32:
        return infinicore.float32
    elif torch_dtype == torch.float16:
        return infinicore.float16
    elif torch_dtype == torch.bfloat16:
        return infinicore.bfloat16
    elif torch_dtype == torch.int8:
        return infinicore.int8
    elif torch_dtype == torch.int16:
        return infinicore.int16
    elif torch_dtype == torch.int32:
        return infinicore.int32
    elif torch_dtype == torch.int64:
        return infinicore.int64
    elif torch_dtype == torch.uint8:
        return infinicore.uint8
    else:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def numpy_to_infinicore_dtype(numpy_dtype):
    """Convert numpy data type to infinicore data type"""
    if numpy_dtype == np.float32:
        return infinicore.float32
    elif numpy_dtype == np.float64:
        return infinicore.float64
    elif numpy_dtype == np.float16:
        return infinicore.float16
    elif hasattr(np, "bfloat16") and numpy_dtype == np.bfloat16:
        return infinicore.bfloat16
    elif ml_dtypes is not None and numpy_dtype == ml_dtypes.bfloat16:
        return infinicore.bfloat16
    elif numpy_dtype == np.int8:
        return infinicore.int8
    elif numpy_dtype == np.int16:
        return infinicore.int16
    elif numpy_dtype == np.int32:
        return infinicore.int32
    elif numpy_dtype == np.int64:
        return infinicore.int64
    elif numpy_dtype == np.uint8:
        return infinicore.uint8
    else:
        raise ValueError(f"Unsupported numpy dtype: {numpy_dtype}")


def infinicore_to_numpy_dtype(infini_dtype):
    """Convert infinicore data type to numpy data type"""
    if infini_dtype == infinicore.float32:
        return np.float32
    elif infini_dtype == infinicore.float64:
        return np.float64
    elif infini_dtype == infinicore.float16:
        return np.float16
    elif infini_dtype == infinicore.int8:
        return np.int8
    elif infini_dtype == infinicore.int16:
        return np.int16
    elif infini_dtype == infinicore.bfloat16:
        if hasattr(np, "bfloat16"):
            return np.bfloat16
        if ml_dtypes is None:
            raise ModuleNotFoundError(
                "ml_dtypes is required for bfloat16 numpy conversion. "
                "Please install ml_dtypes."
            )
        return ml_dtypes.bfloat16
    elif infini_dtype == infinicore.int32:
        return np.int32
    elif infini_dtype == infinicore.int64:
        return np.int64
    elif infini_dtype == infinicore.uint8:
        return np.uint8
    else:
        raise ValueError(f"Unsupported infinicore dtype: {infini_dtype}")
