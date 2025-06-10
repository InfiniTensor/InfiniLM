import gguf
from typing import List
import numpy as np
from gguf import GGMLQuantizationType


def np_dtype_to_ggml(tensor_dtype: np.dtype):
    if tensor_dtype == np.float16:
        return GGMLQuantizationType.F16
    elif tensor_dtype == np.float32:
        return GGMLQuantizationType.F32
    elif tensor_dtype == np.float64:
        return GGMLQuantizationType.F64
    elif tensor_dtype == np.int8:
        return GGMLQuantizationType.I8
    elif tensor_dtype == np.int16:
        return GGMLQuantizationType.I16
    elif tensor_dtype == np.int32:
        return GGMLQuantizationType.I32
    elif tensor_dtype == np.int64:
        return GGMLQuantizationType.I64
    else:
        raise ValueError(
            "Only F16, F32, F64, I8, I16, I32, I64 tensors are supported for now"
        )


def gguf_strides(*args: int) -> list[int] | None:
    return list(args)[::-1] if args else None


def contiguous_gguf_strides(shape: tuple[int, ...]) -> list[int]:
    strides = []
    acc = 1
    for size in reversed(shape):
        strides.append(acc)
        acc *= size
    return strides[::-1]

def process_zero_stride_tensor(tensor, stride=None):
    if stride:
        slices = tuple(slice(0, 1) if s == 0 else slice(None) for s in stride)
        return tensor[slices]
    else:
        return tensor

class InfiniopTestCase:
    op_name: str

    def __init__(self, op_name: str):
        self.op_name = op_name

    def write_test(self, test_writer: "InfiniopTestWriter"):
        test_writer.add_string(test_writer.gguf_key("op_name"), self.op_name)


class InfiniopTestWriter(gguf.GGUFWriter):
    _test_cases: List["InfiniopTestCase"]
    _written_tests = 0

    def __init__(self, filepath):
        super().__init__(filepath, "infiniop-test")
        self._test_cases = []
        self._written_tests = 0

    def add_test(self, test_case: "InfiniopTestCase"):
        self._test_cases.append(test_case)

    def add_tests(self, test_cases: List["InfiniopTestCase"]):
        self._test_cases.extend(test_cases)

    def gguf_key(self, name: str) -> str:
        return f"test.{self._written_tests}.{name}"

    def save(self):
        super().add_uint64("test_count", len(self._test_cases))
        for test_case in self._test_cases:
            test_case.write_test(self)
            self._written_tests += 1
        super().write_header_to_file()
        super().write_kv_data_to_file()
        super().write_tensors_to_file()
        super().close()
