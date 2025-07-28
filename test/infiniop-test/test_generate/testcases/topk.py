from ast import List
import numpy as np
import gguf
from typing import List


from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides


def topk(x: np.ndarray, k: int, axis: int, largest: bool, sorted: bool):
    """
    使用numpy计算topk结果
    Args:
        x:  输入张量
        k: top k
        axis: an int, the axis to sort along
        largest: a bool, to select largest or smallest
        sorted: a bool, whether to sort the results
    Returns:
        输出张量, (values, indices)
    """
    if largest:
        # argsort sorts in ascending order. For largest, we sort a negated array.
        indices = np.argsort(-x, axis=axis)
    else:
        indices = np.argsort(x, axis=axis)

    # Get top k indices
    top_k_indices = np.take(indices, np.arange(k), axis=axis)
    
    # Get top k values
    top_k_values = np.take_along_axis(x, top_k_indices, axis=axis)

    if sorted:
        return top_k_values, top_k_indices.astype(np.int64)
    else:
        # If not sorted, we can just return the values from argsort, which are sorted.
        # The requirement is to return *a* set of top-k values, not necessarily in a random order.
        return top_k_values, top_k_indices.astype(np.int64)

class TopKTestCase(InfiniopTestCase):
    def __init__(
        self,
        x: np.ndarray,
        shape_x: List[int],
        stride_x: List[int] | None,
        values: np.ndarray,
        shape_values: List[int],
        stride_values: List[int] | None,
        indices: np.ndarray,
        shape_indices: List[int],
        stride_indices: List[int] | None,
        k: int,
        axis: int,
        largest: bool,
        sorted: bool,
    ):
        super().__init__("topk")
        self.x = x
        self.shape_x = shape_x
        self.stride_x = stride_x
        self.values = values
        self.shape_values = shape_values
        self.stride_values = stride_values
        self.indices = indices
        self.shape_indices = shape_indices
        self.stride_indices = stride_indices
        self.k = k
        self.axis = axis
        self.largest = largest
        self.sorted = sorted

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        test_writer.add_tensor(
            test_writer.gguf_key("x"), self.x, raw_dtype=np_dtype_to_ggml(self.x.dtype)
        )
        if self.shape_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.shape_x)
        if self.stride_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.stride_x))

        test_writer.add_tensor(
            test_writer.gguf_key("values"), self.values, raw_dtype=np_dtype_to_ggml(self.values.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("indices"), self.indices, raw_dtype=np_dtype_to_ggml(self.indices.dtype)
        )
        
        if self.shape_values is not None:
            test_writer.add_array(test_writer.gguf_key("values.shape"), self.shape_values)
        test_writer.add_array(
            test_writer.gguf_key("values.strides"),
            gguf_strides(*self.stride_values if self.stride_values is not None else contiguous_gguf_strides(self.shape_values))
        )

        if self.shape_indices is not None:
            test_writer.add_array(test_writer.gguf_key("indices.shape"), self.shape_indices)
        test_writer.add_array(
            test_writer.gguf_key("indices.strides"),
            gguf_strides(*self.stride_indices if self.stride_indices is not None else contiguous_gguf_strides(self.shape_indices))
        )
        
        test_writer.add_int32(test_writer.gguf_key("k"), self.k)
        test_writer.add_int32(test_writer.gguf_key("axis"), self.axis)
        test_writer.add_bool(test_writer.gguf_key("largest"), self.largest)
        test_writer.add_bool(test_writer.gguf_key("sorted"), self.sorted)
        
        ans_values, ans_indices = topk(
            self.x.astype(np.float64),
            self.k,
            self.axis,
            self.largest,
            self.sorted,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_values"), ans_values, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_indices"), ans_indices, raw_dtype=gguf.GGMLQuantizationType.I64
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("topk.gguf")
    test_cases = []

    _TEST_CASES_ = [
        # (shape, k, axis, largest, sorted, stride_x, stride_values, stride_indices)
        ((10,), 3, 0, True, True, None, None, None),
        ((10,), 3, 0, False, True, None, None, None),
        ((5, 10), 4, 1, True, True, None, None, None),
        ((5, 10), 4, 1, True, False, None, None, None),
        ((3, 5, 10), 5, 2, True, True, None, None, None),
        ((10, 5, 3), 2, 0, True, True, None, None, None),
        ((10, 5, 3), 4, 1, True, True, None, None, None),
        ((1000,), 100, 0, True, True, None, None, None),
        # With strides
        ((5, 10), 3, 1, True, True, (1, 5), None, None),
    ]

    _TENSOR_DTYPES_ = [np.float16, np.float32]

    for dtype in _TENSOR_DTYPES_:
        for shape, k, axis, largest, sorted, stride_x, stride_values, stride_indices in _TEST_CASES_:
            x = np.random.rand(*shape).astype(dtype)
            
            output_shape = list(shape)
            output_shape[axis] = k
            output_shape = tuple(output_shape)

            values = np.empty(output_shape, dtype=dtype)
            indices = np.empty(output_shape, dtype=np.int64)
            
            test_case = TopKTestCase(
                x=x,
                shape_x=list(shape),
                stride_x=stride_x,
                values=values,
                shape_values=list(output_shape),
                stride_values=stride_values,
                indices=indices,
                shape_indices=list(output_shape),
                stride_indices=stride_indices,
                k=k,
                axis=axis,
                largest=largest,
                sorted=sorted,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save() 