from ast import List
import numpy as np
import gguf
from typing import Optional
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides

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
        shape: tuple,
        atype: np.dtype,
        wtype: np.dtype,
        epsilon: float = 1e-5,
        x_strides: Optional[tuple] = None,
        y_strides: Optional[tuple] = None,
    ):
        super().__init__("rms_norm")
        self.shape = shape
        self.atype = atype
        self.w_shape = (shape[1],)

        if x_strides is not None:
            itemsize = np.dtype(atype).itemsize
            byte_strides = tuple(s * itemsize for s in x_strides)
            max_offset = sum((dim - 1) * abs(stride) for dim, stride in zip(shape, x_strides))
            buffer_size = max_offset + 1
            buffer = np.random.uniform(-1.0, 1.0, buffer_size).astype(atype) * 0.001
            self.x = as_strided(buffer, shape=shape, strides=byte_strides)
        else:
            self.x = random_tensor(shape, atype)

        self.w = random_tensor(self.w_shape, wtype)
        self.epsilon = epsilon
        
        if y_strides is not None:
            itemsize_out = np.dtype(atype).itemsize
            byte_strides_out = tuple(s * itemsize_out for s in y_strides)
            max_offset_out = sum((dim - 1) * abs(stride) for dim, stride in zip(shape, y_strides))
            buffer_size_out = max_offset_out + 1
            buffer_out = np.zeros(buffer_size_out, dtype=atype)
            self.y = as_strided(buffer_out, shape=shape, strides=byte_strides_out)
        else:
            self.y = np.zeros_like(self.x)

        self.ans = rms_norm(self.x.astype(np.float64), self.w.astype(np.float64), self.epsilon)

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_float32(test_writer.gguf_key("epsilon"), self.epsilon)
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
            test_writer.gguf_key("ans"),
            self.ans,
            raw_dtype=np_dtype_to_ggml(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("y"),
            self.y,
            raw_dtype=np_dtype_to_ggml(self.y.dtype),
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("rms_norm.gguf")
    
    test_cases = [
        RMSNormTestCase(
            shape=(2, 256),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            shape=(8, 1024),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            shape=(1, 768),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            shape=(8, 256),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            shape=(500, 4096),
            atype=np.float32,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            shape=(2, 256),
            atype=np.float16,
            wtype=np.float16,
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float16,
            wtype=np.float16,
        ),
        RMSNormTestCase(
            shape=(500, 4096),
            atype=np.float16,
            wtype=np.float16,
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float16,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            shape=(500, 4096),
            atype=np.float16,
            wtype=np.float32,
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float32,
            wtype=np.float32,
            x_strides=(1024, 1),
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float32,
            wtype=np.float32,
            x_strides=(512, 2),
        ),
        RMSNormTestCase(
            shape=(500, 4096),
            atype=np.float32,
            wtype=np.float32,
            x_strides=(9192, 1),
        ),
        RMSNormTestCase(
            shape=(500, 4096),
            atype=np.float32,
            wtype=np.float32,
            x_strides=(4096, 2),
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float16,
            wtype=np.float16,
            x_strides=(1024, 1),
        ),
        RMSNormTestCase(
            shape=(500, 4096),
            atype=np.float16,
            wtype=np.float16,
            x_strides=(9192, 1),
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float16,
            wtype=np.float32,
            x_strides=(1024, 1),
        ),
        RMSNormTestCase(
            shape=(500, 4096),
            atype=np.float16,
            wtype=np.float32,
            x_strides=(9192, 1),
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float32,
            wtype=np.float32,
            y_strides=(1024, 1),
        ),
        RMSNormTestCase(
            shape=(500, 4096),
            atype=np.float32,
            wtype=np.float32,
            y_strides=(8192, 2),
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float32,
            wtype=np.float32,
            x_strides=(1024, 1),
            y_strides=(512, 2),
        ),
        RMSNormTestCase(
            shape=(4, 512),
            atype=np.float16,
            wtype=np.float16,
            y_strides=(2048, 1),
        ),
    ]
    
    test_writer.add_tests(test_cases)
    test_writer.save()
