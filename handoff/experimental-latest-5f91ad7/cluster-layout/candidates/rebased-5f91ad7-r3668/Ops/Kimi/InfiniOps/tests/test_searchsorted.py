import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize("scalar_input", (False, True))
@pytest.mark.parametrize("use_deprecated", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32,))
def test_searchsorted(
    scalar_input,
    use_deprecated,
    dtype,
    device,
    implementation_index,
):
    sorted_sequence = torch.tensor((3.0, 1.0, 2.0, 5.0), dtype=dtype, device=device)
    sorter = torch.tensor((1, 2, 0, 3), dtype=torch.int64, device=device)
    input = (
        2.5
        if scalar_input
        else torch.tensor((0.5, 2.5, 6.0), dtype=dtype, device=device)
    )
    out = torch.empty((), dtype=torch.int64, device=device)

    if not scalar_input:
        out = torch.empty_like(input, dtype=torch.int64)

    args = (sorted_sequence, input)
    attributes = (False, False, None)
    kwargs = {
        "stream": get_stream(device),
        "implementation_index": implementation_index,
    }

    if use_deprecated:
        infini.ops.searchsorted(*args, *attributes, sorter, out, **kwargs)
    else:
        infini.ops.searchsorted(*args, sorter, *attributes, out, **kwargs)

    expected = torch.searchsorted(sorted_sequence, input, sorter=sorter)
    assert torch.equal(out, expected)
