from ast import List
import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def add(
    a: np.ndarray,
    b: np.ndarray,
):
    return a + b


class AddTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        stride_a: List[int] | None,
        b: np.ndarray,
        stride_b: List[int] | None,
        c: np.ndarray,
        stride_c: List[int] | None,
    ):
        super().__init__("add")
        self.a = a
        self.stride_a = stride_a
        self.b = b
        self.stride_b = stride_b
        self.c = c
        self.stride_c = stride_c


    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.stride_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.strides"), self.stride_a)
        if self.stride_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.strides"), self.stride_b)
        if self.stride_c is not None:
            test_writer.add_array(test_writer.gguf_key("c.strides"), self.stride_c)
        test_writer.add_tensor(
            test_writer.gguf_key("a"), self.a, raw_dtype=np_dtype_to_ggml(self.a.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("b"), self.b, raw_dtype=np_dtype_to_ggml(self.b.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("c"), self.c, raw_dtype=np_dtype_to_ggml(self.c.dtype)
        )
        ans = add(
            self.a.astype(np.float64),
            self.b.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("add.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, a_stride, b_stride, c_stride
        ((13, 4), None, None, None),
        ((13, 4), gguf_strides(10, 1), gguf_strides(10, 1), gguf_strides(10, 1)),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), gguf_strides(20, 4, 1), gguf_strides(20, 4, 1), gguf_strides(20, 4, 1)),
        ((16, 5632), None, None, None),
        ((16, 5632), gguf_strides(13312, 1), gguf_strides(13312, 1), gguf_strides(13312, 1)),
        ((4, 4, 5632), None, None, None),
        ((4, 4, 5632), gguf_strides(45056, 5632, 1), gguf_strides(45056, 5632, 1), gguf_strides(45056, 5632, 1)),
    ]
    _TENSOR_DTYPES_ = [np.float16, np.float32]
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_a, stride_b, stride_c in _TEST_CASES_:
            a = np.random.rand(*shape).astype(dtype)
            b = np.random.rand(*shape).astype(dtype)
            c = np.random.rand(*shape).astype(dtype)

            test_case = AddTestCase(
                a=a,
                stride_a=stride_a,
                b=b,
                stride_b=stride_b,
                c=c,
                stride_c=stride_c,
            )
            test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
