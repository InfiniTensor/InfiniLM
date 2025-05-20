import numpy as np
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def swiglu(
    a: np.ndarray,
    b: np.ndarray,
):
    c = a * b / (1.0 + np.exp(-b))

    return c


class SwiGLUTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: np.ndarray,
        stride_a: List[int] | None,
        b: np.ndarray,
        stride_b: List[int] | None,
        c: np.ndarray,
        stride_c: List[int] | None,
    ):
        super().__init__("swiglu")
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
        ans = swiglu(
            self.a.astype(np.float64),
            self.b.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("swiglu.gguf")
    test_cases = [
        SwiGLUTestCase(
            np.random.rand(64, 128).astype(np.float32),
            None,
            np.random.rand(64, 128).astype(np.float32),
            None,
            np.random.rand(64, 128).astype(np.float32),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(64, 121).astype(np.float32),
            None,
            np.random.rand(64, 121).astype(np.float32),
            None,
            np.random.rand(64, 121).astype(np.float32),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(15, 512).astype(np.float32),
            None,
            np.random.rand(15, 512).astype(np.float32),
            None,
            np.random.rand(15, 512).astype(np.float32),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4).astype(np.float32),
            None,
            np.random.rand(13, 4).astype(np.float32),
            None,
            np.random.rand(13, 4).astype(np.float32),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4).astype(np.float16),
            None,
            np.random.rand(13, 4).astype(np.float16),
            None,
            np.random.rand(13, 4).astype(np.float16),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4).astype(np.float32),
            gguf_strides(10, 1),
            np.random.rand(13, 4).astype(np.float32),
            gguf_strides(10, 1),
            np.random.rand(13, 4).astype(np.float32),
            gguf_strides(10, 1),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4).astype(np.float16),
            gguf_strides(10, 1),
            np.random.rand(13, 4).astype(np.float16),
            gguf_strides(10, 1),
            np.random.rand(13, 4).astype(np.float16),
            gguf_strides(10, 1),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4, 4).astype(np.float32),
            None,
            np.random.rand(13, 4, 4).astype(np.float32),
            None,
            np.random.rand(13, 4, 4).astype(np.float32),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4, 4).astype(np.float16),
            None,
            np.random.rand(13, 4, 4).astype(np.float16),
            None,
            np.random.rand(13, 4, 4).astype(np.float16),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4, 4).astype(np.float32),
            gguf_strides(20, 4, 1),
            np.random.rand(13, 4, 4).astype(np.float32),
            gguf_strides(20, 4, 1),
            np.random.rand(13, 4, 4).astype(np.float32),
            gguf_strides(20, 4, 1),
        ),
        SwiGLUTestCase(
            np.random.rand(13, 4, 4).astype(np.float16),
            gguf_strides(20, 4, 1),
            np.random.rand(13, 4, 4).astype(np.float16),
            gguf_strides(20, 4, 1),
            np.random.rand(13, 4, 4).astype(np.float16),
            gguf_strides(20, 4, 1),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float32),
            None,
            np.random.rand(16, 5632).astype(np.float32),
            None,
            np.random.rand(16, 5632).astype(np.float32),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float16),
            None,
            np.random.rand(16, 5632).astype(np.float16),
            None,
            np.random.rand(16, 5632).astype(np.float16),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(13312, 1),
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(13312, 1),
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(13312, 1),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(13312, 1),
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(13312, 1),
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(13312, 1),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(5632, 1),
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(5632, 1),
            np.random.rand(16, 5632).astype(np.float32),
            gguf_strides(1, 16),
        ),
        SwiGLUTestCase(
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(5632, 1),
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(5632, 1),
            np.random.rand(16, 5632).astype(np.float16),
            gguf_strides(1, 16),
        ),
        SwiGLUTestCase(
            np.random.rand(2, 3, 400).astype(np.float32),
            gguf_strides(1200, 400, 1),
            np.random.rand(2, 3, 400).astype(np.float32),
            gguf_strides(1200, 400, 1),
            np.random.rand(2, 3, 400).astype(np.float32),
            gguf_strides(1, 2, 6),
        ),
        SwiGLUTestCase(
            np.random.rand(2, 3, 400).astype(np.float16),
            gguf_strides(1200, 400, 1),
            np.random.rand(2, 3, 400).astype(np.float16),
            gguf_strides(1200, 400, 1),
            np.random.rand(2, 3, 400).astype(np.float16),
            gguf_strides(1, 2, 6),
        ),
        SwiGLUTestCase(
            np.random.rand(4, 4, 5632).astype(np.float32),
            None,
            np.random.rand(4, 4, 5632).astype(np.float32),
            None,
            np.random.rand(4, 4, 5632).astype(np.float32),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(4, 4, 5632).astype(np.float16),
            None,
            np.random.rand(4, 4, 5632).astype(np.float16),
            None,
            np.random.rand(4, 4, 5632).astype(np.float16),
            None,
        ),
        SwiGLUTestCase(
            np.random.rand(4, 4, 5632).astype(np.float32),
            gguf_strides(45056, 5632, 1),
            np.random.rand(4, 4, 5632).astype(np.float32),
            gguf_strides(45056, 5632, 1),
            np.random.rand(4, 4, 5632).astype(np.float32),
            gguf_strides(45056, 5632, 1),
        ),
        SwiGLUTestCase(
            np.random.rand(4, 4, 5632).astype(np.float16),
            gguf_strides(45056, 5632, 1),
            np.random.rand(4, 4, 5632).astype(np.float16),
            gguf_strides(45056, 5632, 1),
            np.random.rand(4, 4, 5632).astype(np.float16),
            gguf_strides(45056, 5632, 1),
        ),
    ]
    test_writer.add_tests(test_cases)
    test_writer.save()
