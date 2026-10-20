import infini.ops
import pytest
import torch

from tests.utils import get_stream


@pytest.mark.parametrize(
    "dtype",
    (torch.float32, torch.float16, torch.bfloat16),
)
def test_argmax_flattened(dtype, device, implementation_index):
    if device != "cuda":
        pytest.skip("argmax requires a CUDA-family backend")

    input = torch.randn(32_003, dtype=dtype, device=device)
    input[17_391] = 100
    out = torch.full((), -1, dtype=torch.int64, device=device)

    result = infini.ops.argmax(
        input,
        None,
        False,
        out,
        stream=get_stream(input.device),
        implementation_index=implementation_index,
    )

    assert result is None
    assert out.item() == torch.argmax(input).item()


def test_argmax_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default streams require a CUDA-family backend")

    input = torch.randn(32_003, dtype=torch.float16, device=device)
    input[17_391] = 100
    out = torch.full((), -1, dtype=torch.int64, device=device)
    expected = torch.argmax(input).item()
    torch.cuda.synchronize()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    # If the provider ignores the explicit stream, this delay keeps its work
    # from completing before the requested stream is synchronized.
    torch.cuda._sleep(50_000_000)
    try:
        infini.ops.argmax(
            input,
            None,
            False,
            out,
            stream=stream.cuda_stream,
            implementation_index=implementation_index,
        )
        stream.synchronize()
        with torch.cuda.stream(stream):
            actual = out.cpu().item()
        assert actual == expected
    finally:
        torch.cuda.synchronize()
