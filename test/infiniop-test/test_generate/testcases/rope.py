from ast import List
import numpy as np
import gguf
from typing import List


from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides


def rotary_embedding(t, sin, cos):
    dh = t.shape[2] 
    assert dh % 2 == 0, "Embedding dimension must be even."

    t_even = t[..., 0::2]  # [seq_len, n_head, dh // 2]
    t_odd = t[..., 1::2]  # [seq_len, n_head, dh // 2]

    cos = np.expand_dims(cos, axis=1)  # [seq_len, 1, dh // 2]
    sin = np.expand_dims(sin, axis=1)  # [seq_len, 1, dh // 2]

    t_out_even = t_even * cos - t_odd * sin
    t_out_odd = t_even * sin + t_odd * cos

    t_out = np.empty_like(t)
    t_out[..., 0::2] = t_out_even
    t_out[..., 1::2] = t_out_odd

    return t_out


def sin_cos_table(pos, dim, theta, dtype):
    assert dim % 2 == 0, "Embedding dimension must be even."

    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))

    angles = np.outer(pos, freqs)

    sin_vals = np.sin(angles).astype(dtype)
    cos_vals = np.cos(angles).astype(dtype)

    return sin_vals, cos_vals


class RoPETestCase(InfiniopTestCase):
    def __init__(
        self,
        y: np.ndarray,
        x: np.ndarray,
        shape_y: List[int] | None,
        shape_x: List[int] | None,
        stride_y: List[int] | None,
        stride_x: List[int] | None,
        pos_ids: np.ndarray,
        sin_table: np.ndarray,
        cos_table: np.ndarray,
    ):
        super().__init__("rope")
        self.y = y
        self.x = x
        self.shape_y = shape_y
        self.shape_x = shape_x
        self.stride_y = stride_y
        self.stride_x = stride_x
        self.pos_ids = pos_ids
        self.sin_table = sin_table
        self.cos_table = cos_table

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        test_writer.add_tensor(
            test_writer.gguf_key("y"), self.y, raw_dtype=np_dtype_to_ggml(self.y.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("x"), self.x, raw_dtype=np_dtype_to_ggml(self.x.dtype)
        )
        if self.shape_y is not None:
            test_writer.add_array(test_writer.gguf_key("y.shape"), self.shape_y)
        if self.shape_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.shape_x)
        test_writer.add_array(
            test_writer.gguf_key("y.strides"),
            gguf_strides(*self.stride_y if self.stride_y is not None else contiguous_gguf_strides(self.shape_y))
        )
        if self.stride_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.stride_x))

        test_writer.add_tensor(
            test_writer.gguf_key("pos_ids"), self.pos_ids, raw_dtype=np_dtype_to_ggml(self.pos_ids.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("sin_table"), self.sin_table, raw_dtype=np_dtype_to_ggml(self.sin_table.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("cos_table"), self.cos_table, raw_dtype=np_dtype_to_ggml(self.cos_table.dtype)
        )
        ans = rotary_embedding(
            self.x.astype(np.float64),
            self.sin_table.astype(np.float64),
            self.cos_table.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )




if __name__ == "__main__":
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # (shape, x_strides, y_strides)
        ((1, 32, 128), None, None),
        ((10, 32, 64), None, None),
        # # 昇腾暂不满足这个用例，最后一维度 <=32 会有问题，可能与其核心
        # # 接口 GatherMask 的内部实现相关，目前 48 64 128 都可以支持
        ((4, 1, 32), (64, 64, 1), None),
        ((11, 33, 128), None, (8000, 200, 1)),
        ((3, 32, 128), (8000, 200, 1), (7000, 128, 1)),
    ]

    _TENSOR_DTYPES_ = [np.float16, np.float32]
    test_writer = InfiniopTestWriter("rope.gguf")
    test_cases = []

    for dtype in _TENSOR_DTYPES_:
        for shape, stride_x, stride_y in _TEST_CASES_:
            x = np.random.rand(*shape).astype(dtype)
            y = np.empty(tuple(0 for _ in shape), dtype=dtype)
            pos_ids = np.arange(0, x.shape[0], dtype=np.int32)
            sin_table, cos_table = sin_cos_table(pos_ids, x.shape[2], theta=1e5, dtype=dtype)
            test_case = RoPETestCase(
                y=y,
                x=x,
                shape_y=shape,
                shape_x=shape,
                stride_y=stride_y,
                stride_x=stride_x,
                pos_ids=pos_ids,
                sin_table=sin_table,
                cos_table=cos_table,
            )
            test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
