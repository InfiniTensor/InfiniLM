import numpy as np
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def random_tensor(shape: tuple, dtype: np.dtype) -> np.ndarray:
    return np.random.uniform(-1.0, 1.0, shape).astype(dtype) * 0.001

def rms_norm(x: np.ndarray, w: np.ndarray, epsilon: float) -> np.ndarray:
    """
    使用numpy计算rms_norm结果
    Args:
        x:  输入张量, 维度为2, 形状为 [..., hidden_size]
        w: 缩放权重, 形状为 [hidden_size]
        epsilon: 避免除零的小常数
    Returns:
        输出张量, 形状与 input 相同
    """
    squared = x ** 2
    mean = np.mean(squared, axis=-1, keepdims=True)
    rms = np.sqrt(mean + epsilon)
    
    normalized = x / rms
    return normalized * w

class RMSNormTestCase(InfiniopTestCase):
    def __init__(
        self,
        x: np.ndarray,
        w: np.ndarray,
        y: np.ndarray,
        shape: List[int] | None,
        x_strides: List[int] | None,
        y_strides: List[int] | None,
        epsilon: float = 1e-5,
    ):
        super().__init__("rms_norm")
        self.x = x
        self.w = w
        self.y = y
        self.shape = shape
        self.epsilon = epsilon
        self.x_strides=x_strides
        self.y_strides=y_strides
        
    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_float32(test_writer.gguf_key("epsilon"), self.epsilon)
        if self.shape is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.shape)
            test_writer.add_array(test_writer.gguf_key("y.shape"), self.shape)
        if self.x_strides is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.x_strides))
        test_writer.add_array(
            test_writer.gguf_key("y.strides"),
            gguf_strides(*self.y_strides if self.y_strides is not None else contiguous_gguf_strides(self.shape))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("x"),
            self.x,
            raw_dtype=np_dtype_to_ggml(self.x.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("w"),
            self.w,
            raw_dtype=np_dtype_to_ggml(self.w.dtype),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("y"),
            self.y,
            raw_dtype=np_dtype_to_ggml(self.y.dtype),
        )
        ans = rms_norm(self.x.astype(np.float64), self.w.astype(np.float64), self.epsilon)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans,
            raw_dtype=np_dtype_to_ggml(np.float64),
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("rms_norm.gguf")
    test_cases = []

    _TEST_CASES_ = [
        # shape, x_strides, y_strides
        ((2, 256), None, None),
        ((4, 512), None, None),
        ((8, 1024), None, None),
        ((1, 768), None, None),
        ((8, 256), None, None),
        ((500, 4096), None, None),
        ((4, 512), (1024, 1), None),
        ((4, 512), (512, 1), None),
        ((500, 4096), (9192, 1), None),
        ((500, 4096), (4096, 1), None),
        ((4, 512), None, (1024, 1)),
        ((500, 4096), None, (8192, 1)),
        ((4, 512), (1024, 1), (512, 1)),
        ((4, 512), None, (2048, 1)),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, x_strides, y_strides in _TEST_CASES_:
            w = np.random.rand(shape[-1]).astype(dtype)
            x = np.random.rand(*shape).astype(dtype)
            y = np.empty(tuple(0 for _ in shape), dtype=dtype)
            epsilon = 1e-5
            test_case = RMSNormTestCase(
                x=x,
                w=w,
                y=y,
                shape=shape,
                x_strides=x_strides,
                y_strides=y_strides,
                epsilon=epsilon
            )
            test_cases.append(test_case)        

    test_writer.add_tests(test_cases)
    test_writer.save()
