import numpy as np
import gguf
from typing import List, Optional, Tuple

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides


def clip(
    x: np.ndarray,
    min_val: np.ndarray,
    max_val: np.ndarray,
) -> np.ndarray:
    """
    Clip the values in input tensor x to the range [min_val, max_val].

    Args:
        x: Input tensor
        min_val: Tensor with minimum values (same shape as x)
        max_val: Tensor with maximum values (same shape as x)

    Returns:
        Clipped tensor with the same shape as x
    """
    return np.maximum(np.minimum(x, max_val), min_val)


def random_tensor(shape, dtype):
    """
    Generate a random tensor with values in the range [-2, 2].

    Args:
        shape: Shape of the tensor
        dtype: Data type of the tensor

    Returns:
        Random tensor with the specified shape and dtype
    """
    return (np.random.rand(*shape).astype(dtype) * 4.0 - 2.0)


class ClipTestCase(InfiniopTestCase):
    """
    Test case for the Clip operator.
    """

    def __init__(
        self,
        x: np.ndarray,
        x_stride: Optional[List[int]],
        min_val: np.ndarray,
        min_stride: Optional[List[int]],
        max_val: np.ndarray,
        max_stride: Optional[List[int]],
        y: np.ndarray,
        y_shape:  Optional[List[int]],
        y_stride: Optional[List[int]],
    ):
        super().__init__("clip")
        self.x = x
        self.x_stride = x_stride
        self.min_val = min_val
        self.min_stride = min_stride
        self.max_val = max_val
        self.max_stride = max_stride
        self.y = y
        self.y_shape=y_shape
        self.y_stride = y_stride

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # Add strides as arrays if they exist
        if self.x_stride is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.x_stride))
        if self.min_stride is not None:
            test_writer.add_array(test_writer.gguf_key("min_val.strides"), gguf_strides(*self.min_stride))
        if self.max_stride is not None:
            test_writer.add_array(test_writer.gguf_key("max_val.strides"), gguf_strides(*self.max_stride))
        if self.y_shape is not None:
            test_writer.add_array(test_writer.gguf_key("y.shape"), self.y_shape)
        test_writer.add_array(
            test_writer.gguf_key("y.strides"),
            gguf_strides(*self.y_stride if self.y_stride is not None else contiguous_gguf_strides(self.y_shape))
        )

        # Add tensors to the test
        test_writer.add_tensor(
            test_writer.gguf_key("x"),
            self.x,
            raw_dtype=np_dtype_to_ggml(self.x.dtype)
        )

        test_writer.add_tensor(
            test_writer.gguf_key("min_val"),
            self.min_val,
            raw_dtype=np_dtype_to_ggml(self.min_val.dtype)
        )

        test_writer.add_tensor(
            test_writer.gguf_key("max_val"),
            self.max_val,
            raw_dtype=np_dtype_to_ggml(self.max_val.dtype)
        )

        test_writer.add_tensor(
            test_writer.gguf_key("y"),
            self.y,
            raw_dtype=np_dtype_to_ggml(self.y.dtype)
        )

        # Calculate the expected result
        ans = clip(
            self.x.astype(np.float64),
            self.min_val.astype(np.float64),
            self.max_val.astype(np.float64)
        )

        # Add the expected result to the test
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans,
            raw_dtype=gguf.GGMLQuantizationType.F64
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("clip.gguf")

    # Create test cases for different shapes, strides, and data types
    test_cases = []

    # Test case shapes
    shapes = [
        (10,),                # 1D tensor
        (5, 10),              # 2D tensor
        (2, 3, 4),            # 3D tensor
        (7, 13),              # Prime dimensions
        (1, 1),               # Minimum shape
        (100, 100),           # Large shape
        (16, 16, 16),         # Large 3D
    ]

    # Test case min/max values
    min_max_values = [
        (-1.0, 1.0),          # Standard range
        (0.0, 2.0),           # Positive range
        (-2.0, 0.0),          # Negative range
        (-1000.0, 1000.0),    # Large range
        (-0.001, 0.001),      # Small range
        (0.0, 0.0),           # min=max
    ]

    # Data types to test
    dtypes = [np.float16, np.float32, np.float64]

    # Generate test cases with contiguous tensors
    for shape in shapes:
        for min_val, max_val in min_max_values:
            for dtype in dtypes:
                x = random_tensor(shape, dtype)
                min_tensor = np.full(shape, min_val, dtype=dtype)
                max_tensor = np.full(shape, max_val, dtype=dtype)
                y = np.empty(tuple(0 for _ in shape), dtype=dtype)

                test_cases.append(
                    ClipTestCase(
                        x=x,
                        x_stride=None,
                        min_val=min_tensor,
                        min_stride=None,
                        max_val=max_tensor,
                        max_stride=None,
                        y=y,
                        y_shape=shape,
                        y_stride=None
                    )
                )

    # Generate test cases with strided tensors (for 2D shapes only)
    for shape in [s for s in shapes if len(s) == 2]:
        for dtype in dtypes:
            # Row-major stride
            row_stride = (shape[1], 1)
            # Column-major stride
            col_stride = (1, shape[0])

            # Test case with row-major input and output
            x = random_tensor(shape, dtype)
            min_tensor = np.full(shape, -1.0, dtype=dtype)
            max_tensor = np.full(shape, 1.0, dtype=dtype)
            y = np.empty(tuple(0 for _ in shape), dtype=dtype)

            test_cases.append(
                ClipTestCase(
                    x=x,
                    x_stride=row_stride,
                    min_val=min_tensor,
                    min_stride=row_stride,
                    max_val=max_tensor,
                    max_stride=row_stride,
                    y=y,
                    y_shape=shape,
                    y_stride=row_stride
                )
            )

            # Test case with column-major input and output
            x = random_tensor(shape, dtype)
            min_tensor = np.full(shape, -1.0, dtype=dtype)
            max_tensor = np.full(shape, 1.0, dtype=dtype)
            y = np.empty(tuple(0 for _ in shape), dtype=dtype)

            test_cases.append(
                ClipTestCase(
                    x=x,
                    x_stride=col_stride,
                    min_val=min_tensor,
                    min_stride=col_stride,
                    max_val=max_tensor,
                    max_stride=col_stride,
                    y=y,
                    y_shape=shape,
                    y_stride=col_stride
                )
            )

            # Test case with different strides for input and output
            x = random_tensor(shape, dtype)
            min_tensor = np.full(shape, -1.0, dtype=dtype)
            max_tensor = np.full(shape, 1.0, dtype=dtype)
            y = np.empty(tuple(0 for _ in shape), dtype=dtype)

            test_cases.append(
                ClipTestCase(
                    x=x,
                    x_stride=row_stride,
                    min_val=min_tensor,
                    min_stride=row_stride,
                    max_val=max_tensor,
                    max_stride=row_stride,
                    y=y,
                    y_shape=shape,
                    y_stride=col_stride
                )
            )

    # Add all test cases to the writer
    test_writer.add_tests(test_cases)

    # Save the test cases to a GGUF file
    test_writer.save()

    print(f"Generated {len(test_cases)} test cases for the Clip operator")
