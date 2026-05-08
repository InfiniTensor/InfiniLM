import numpy as np

import infinilm.core as infinicore


def test_tensor_to_numpy_returns_cpu_copy():
    tensor = infinicore.from_list([[1, 2], [3, 4]], dtype=infinicore.int64)

    array = tensor.to_numpy()

    np.testing.assert_array_equal(array, np.array([[1, 2], [3, 4]], dtype=np.int64))
    assert array.shape == (2, 2)
    array[0, 0] = 99
    assert tensor.to_numpy()[0, 0] == 1


def test_generation_utils_does_not_override_tensor_to_numpy():
    import infinilm.generation.utils  # noqa: F401

    tensor = infinicore.from_list([1.5], dtype=infinicore.float16)

    array = tensor.to_numpy()

    np.testing.assert_array_equal(array, np.array([1.5], dtype=np.float16))
