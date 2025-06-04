import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

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
        shape_a: List[int] | None,        
        stride_a: List[int] | None,
        b: np.ndarray,
        shape_b: List[int] | None,       
        stride_b: List[int] | None,
        c: np.ndarray,
        shape_c: List[int] | None,    
        stride_c: List[int] | None,
    ):
        super().__init__("mul")
        self.a = a
        self.shape_a = shape_a
        self.stride_a = stride_a
        self.b = b
        self.shape_b = shape_b
        self.stride_b = stride_b
        self.c = c
        self.shape_c = shape_c
        self.stride_c = stride_c


    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.shape_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.shape"), self.shape_a)
        if self.shape_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.shape"), self.shape_b)
        if self.shape_c is not None:
            test_writer.add_array(test_writer.gguf_key("c.shape"), self.shape_c)
        if self.stride_a is not None:
            test_writer.add_array(test_writer.gguf_key("a.strides"), gguf_strides(*self.stride_a))
        if self.stride_b is not None:
            test_writer.add_array(test_writer.gguf_key("b.strides"), gguf_strides(*self.stride_b))
        test_writer.add_array(
            test_writer.gguf_key("c.strides"),
            gguf_strides(*self.stride_c if self.stride_c is not None else contiguous_gguf_strides(self.shape_c))
        )

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
    test_cases = []

    _TEST_CASES_ = [
        ((2, 3), (3, 1), (1, 2), (3, 1)),
        ((2, 3), (1, 2), (3, 1), (1, 2)),
        ((2, 3), (3, 1), (3, 1), (1, 2)),
        ((4, 6), (1, 4), (1, 5), (6, 1)),
        ((1, 2048), (1, 1), (2048, 1), (1, 1)),
        ((2048, 2048), None, (1, 2048), None),
        ((2, 4, 2048), (4 * 2048, 2048, 1), (1, 2, 8), (4 * 2048, 2048, 1)),
        ((2, 4, 2048), (1, 2, 8), None, (1, 2, 8)),
        ((2048, 2560), (2560, 1), (1, 2048), (2560, 1)),
        ((4, 48, 64), (64 * 48, 64, 1), (1, 4, 192), None),
        ((4, 48, 64), None, (1, 4, 192), (48 * 64, 64, 1)),
    ]   
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_a, stride_b, stride_c in _TEST_CASES_:
            a = random_tensor(shape, dtype)
            b = random_tensor(shape, dtype)
            c = np.empty(tuple(0 for _ in shape), dtype=dtype)

                
            test_cases.append(
                MulTestCase(
                    a=a,
                    shape_a=shape,
                    stride_a=stride_a,
                    b=b,
                    shape_b=shape,
                    stride_b=stride_b,
                    c=c,
                    shape_c=shape,
                    stride_c=stride_c,
                )
            )   
    
    test_writer.add_tests(test_cases)
    test_writer.save()
