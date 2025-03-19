from ast import List
import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def matmul(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 1.0,
    c: np.ndarray = None,
    beta: float = 0.0,
):
    if c is None:
        return alpha * np.matmul(a, b)
    return alpha * np.matmul(a, b) + beta * c


class MatmulTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        stride_a: List[int] | None,
        b: np.ndarray,
        stride_b: List[int] | None,
        c: np.ndarray,
        stride_c: List[int] | None,
        alpha: float,
        beta: float,
    ):
        super().__init__("matmul")
        self.a = a
        self.stride_a = stride_a
        self.b = b
        self.stride_b = stride_b
        self.c = c
        self.stride_c = stride_c
        self.alpha = alpha
        self.beta = beta

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.stride_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.strides"), self.stride_a)
        if self.stride_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.strides"), self.stride_b)
        if self.stride_c is not None:
            test_writer.add_array(test_writer.gguf_key("c.strides"), self.stride_c)
        test_writer.add_float32(test_writer.gguf_key("alpha"), self.alpha)
        test_writer.add_float32(test_writer.gguf_key("beta"), self.beta)
        test_writer.add_tensor(
            test_writer.gguf_key("a"), self.a, raw_dtype=np_dtype_to_ggml(self.a.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("b"), self.b, raw_dtype=np_dtype_to_ggml(self.b.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("c"), self.c, raw_dtype=np_dtype_to_ggml(self.c.dtype)
        )
        ans = matmul(
            self.a.astype(np.float64),
            self.b.astype(np.float64),
            self.alpha,
            self.c.astype(np.float64),
            self.beta,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("matmul.gguf")
    # a, stride_a, b, stride_b, c, stride_c, alpha, beta
    test_cases = [
        MatmulTestCase(
            np.random.rand(4, 5).astype(np.float32),
            None,
            np.random.rand(5, 6).astype(np.float32),
            None,
            np.random.rand(4, 6).astype(np.float32),
            None,
            1.0,
            0.0,
        ),
        MatmulTestCase(
            np.random.rand(4, 5).astype(np.float32),
            gguf_strides(1, 4),
            np.random.rand(5, 6).astype(np.float32),
            gguf_strides(1, 5),
            np.random.rand(4, 6).astype(np.float32),
            gguf_strides(1, 4),
            1.0,
            1.0,
        ),
    ]
    test_writer.add_tests(test_cases)
    test_writer.save()
