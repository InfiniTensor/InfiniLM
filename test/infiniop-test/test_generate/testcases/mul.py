import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides

def mul(
    a: np.ndarray,
    b: np.ndarray
):
    return np.multiply(a, b)

def random_tensor(shape, dtype):
    rate = 1e-3
    var = 0.5 * rate  # 数值范围在[-5e-4, 5e-4]
    return rate * np.random.rand(*shape).astype(dtype) - var

class MulTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        stride_a: List[int] | None,
        b: np.ndarray,
        stride_b: List[int] | None,
        c: np.ndarray,
        stride_c: List[int] | None,
    ):
        super().__init__("mul")
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
        a_fp64 = self.a.astype(np.float64)
        b_fp64 = self.b.astype(np.float64)
        ans_fp64 = np.multiply(a_fp64, b_fp64)
        ans = mul(self.a, self.b)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=np_dtype_to_ggml(ans.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_fp64"),
            ans_fp64,
            raw_dtype=np_dtype_to_ggml(ans_fp64.dtype),
        )

if __name__ == '__main__':
    test_writer = InfiniopTestWriter("mul.gguf")
    test_cases = [
        MulTestCase(
            random_tensor((2, 3), np.float32),
            gguf_strides(3, 1),  
            random_tensor((2, 3), np.float32),
            gguf_strides(1, 2),  
            random_tensor((2, 3), np.float32),
            gguf_strides(3, 1),  
        ),
        MulTestCase(
            random_tensor((2, 3), np.float16),
            gguf_strides(1, 2),  
            random_tensor((2, 3), np.float16),
            gguf_strides(3, 1), 
            random_tensor((2, 3), np.float16),
            gguf_strides(1, 2),  
        ),
        MulTestCase(
            random_tensor((2, 3), np.float64),
            gguf_strides(3, 1),  
            random_tensor((2, 3), np.float64),
            gguf_strides(3, 1),  
            random_tensor((2, 3), np.float64),
            gguf_strides(1, 2),  
        ),
        MulTestCase(
            random_tensor((4, 6), np.float16),
            gguf_strides(1, 4),  
            random_tensor((4, 6), np.float16),
            gguf_strides(1, 5),  
            random_tensor((4, 6), np.float16),
            gguf_strides(6, 1),  
        ),
        MulTestCase(
            random_tensor((1, 2048), np.float16),
            gguf_strides(1, 1),  
            random_tensor((1, 2048), np.float16),
            gguf_strides(2048, 1),  
            random_tensor((1, 2048), np.float16),
            gguf_strides(1, 1),  
        ),
        MulTestCase(
            random_tensor((2048, 2048), np.float32),
            None,  
            random_tensor((2048, 2048), np.float32),
            gguf_strides(1, 2048),  
            random_tensor((2048, 2048), np.float32),
            None,  
        ),
        MulTestCase(
            random_tensor((2, 4, 2048), np.float16),
            gguf_strides(4 * 2048, 2048, 1),  
            random_tensor((2, 4, 2048), np.float16),
            gguf_strides(1, 2, 2 * 4),  
            random_tensor((2, 4, 2048), np.float16),
            gguf_strides(4 * 2048, 2048, 1),  
        ),
        MulTestCase(
            random_tensor((2, 4, 2048), np.float32),
            gguf_strides(1, 2, 2 * 4),  
            random_tensor((2, 4, 2048), np.float32),
            None,  
            random_tensor((2, 4, 2048), np.float32),
            gguf_strides(1, 2, 2 * 4),  
        ),
        MulTestCase(
            random_tensor((2048, 2560), np.float32),
            gguf_strides(2560, 1),  
            random_tensor((2048, 2560), np.float32),
            gguf_strides(1, 2048),  
            random_tensor((2048, 2560), np.float32),
            gguf_strides(2560, 1),  
        ),
        MulTestCase(
            random_tensor((4, 48, 64), np.float16),
            gguf_strides(64 * 48, 64, 1),  
            random_tensor((4, 48, 64), np.float16),
            gguf_strides(1, 4, 4 * 48),  
            random_tensor((4, 48, 64), np.float16),
            None  
        ),
        MulTestCase(
            random_tensor((4, 48, 64), np.float32),
            None,  
            random_tensor((4, 48, 64), np.float32),
            gguf_strides(1, 4, 4 * 48),  
            random_tensor((4, 48, 64), np.float32),
            gguf_strides(48 * 64, 64, 1),  
        )
    ]
    test_writer.add_tests(test_cases)
    test_writer.save()
