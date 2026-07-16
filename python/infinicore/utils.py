def to_torch_dtype(value):
    import torch

    import infinicore

    mapping = {
        infinicore.int8: torch.int8,
        infinicore.int16: torch.int16,
        infinicore.int32: torch.int32,
        infinicore.int64: torch.int64,
        infinicore.uint8: torch.uint8,
        infinicore.float16: torch.float16,
        infinicore.bfloat16: torch.bfloat16,
        infinicore.float32: torch.float32,
        infinicore.float64: torch.float64,
    }
    for name in ("uint16", "uint32", "uint64"):
        torch_dtype = getattr(torch, name, None)
        if torch_dtype is not None:
            mapping[getattr(infinicore, name)] = torch_dtype
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unsupported infinicore dtype: {value}") from error


def to_infinicore_dtype(value):
    import torch

    import infinicore

    mapping = {
        torch.int8: infinicore.int8,
        torch.int16: infinicore.int16,
        torch.int32: infinicore.int32,
        torch.int64: infinicore.int64,
        torch.uint8: infinicore.uint8,
        torch.float16: infinicore.float16,
        torch.bfloat16: infinicore.bfloat16,
        torch.float32: infinicore.float32,
        torch.float64: infinicore.float64,
    }
    for name in ("uint16", "uint32", "uint64"):
        torch_dtype = getattr(torch, name, None)
        if torch_dtype is not None:
            mapping[torch_dtype] = getattr(infinicore, name)
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unsupported torch dtype: {value}") from error


def numpy_to_infinicore_dtype(value):
    import numpy as np

    import infinicore

    mapping = {
        np.dtype("int8"): infinicore.int8,
        np.dtype("int16"): infinicore.int16,
        np.dtype("int32"): infinicore.int32,
        np.dtype("int64"): infinicore.int64,
        np.dtype("uint8"): infinicore.uint8,
        np.dtype("uint16"): infinicore.uint16,
        np.dtype("uint32"): infinicore.uint32,
        np.dtype("uint64"): infinicore.uint64,
        np.dtype("float16"): infinicore.float16,
        np.dtype("float32"): infinicore.float32,
        np.dtype("float64"): infinicore.float64,
    }
    try:
        return mapping[np.dtype(value)]
    except KeyError as error:
        raise ValueError(f"unsupported numpy dtype: {value}") from error


def infinicore_to_numpy_dtype(value):
    import numpy as np

    import infinicore

    mapping = {
        infinicore.int8: np.dtype("int8"),
        infinicore.int16: np.dtype("int16"),
        infinicore.int32: np.dtype("int32"),
        infinicore.int64: np.dtype("int64"),
        infinicore.uint8: np.dtype("uint8"),
        infinicore.uint16: np.dtype("uint16"),
        infinicore.uint32: np.dtype("uint32"),
        infinicore.uint64: np.dtype("uint64"),
        infinicore.float16: np.dtype("float16"),
        infinicore.float32: np.dtype("float32"),
        infinicore.float64: np.dtype("float64"),
    }
    try:
        return mapping[value]
    except KeyError as error:
        raise ValueError(f"unsupported infinicore dtype: {value}") from error
