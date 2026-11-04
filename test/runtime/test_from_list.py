import numpy as np
import pytest

import infinilm._core as infinicore


def test_copy_from_list_updates_existing_tensor():
    tensor = infinicore.from_list([[1, 2], [3, 4]], dtype=infinicore.int32)

    returned = tensor.copy_from_list_([[5, 6], [7, 8]])

    assert returned is tensor
    np.testing.assert_array_equal(
        tensor.to_numpy(), np.array([[5, 6], [7, 8]], dtype=np.int32)
    )


def test_copy_from_list_rejects_shape_change():
    tensor = infinicore.from_list([1, 2], dtype=infinicore.int64)

    with pytest.raises(ValueError, match="input shape"):
        tensor.copy_from_list_([[1, 2]])
