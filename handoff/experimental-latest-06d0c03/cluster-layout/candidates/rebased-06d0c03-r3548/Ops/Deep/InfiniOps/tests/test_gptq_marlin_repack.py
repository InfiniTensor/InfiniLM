# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Reference permutation adapted from vLLM at commit
# `ffc4f08c8ee130d4ea6347c1bf31ffd4f8af28ab`:
# `vllm/model_executor/layers/quantization/utils/marlin_utils_test.py`.

import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "GptqMarlinRepack"):
    pytest.skip(
        "`GptqMarlinRepack` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    ("num_bits", "is_a_8bit", "has_perm"),
    (
        (4, False, False),
        (4, False, True),
        (8, False, False),
        (8, False, True),
        (4, True, False),
        (8, True, False),
    ),
)
@pytest.mark.parametrize(("size_k", "size_n"), ((128, 64), (256, 128)))
def test_gptq_marlin_repack(
    num_bits,
    is_a_8bit,
    has_perm,
    size_k,
    size_n,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("`gptq_marlin_repack` requires the NVIDIA backend")

    unpacked = _make_unpacked_weight(size_k, size_n, num_bits, device)
    b_q_weight = _gptq_pack(unpacked, num_bits)
    perm = _make_perm(size_k, device) if has_perm else _empty_perm(device)
    expected_unpacked = unpacked[perm.long()] if has_perm else unpacked
    expected = _marlin_pack(expected_unpacked, num_bits, is_a_8bit)
    out = torch.empty_like(expected)

    _gptq_marlin_repack(
        b_q_weight,
        perm,
        size_k,
        size_n,
        num_bits,
        is_a_8bit,
        out,
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_gptq_marlin_repack_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    size_k, size_n, num_bits = 128, 64, 4
    unpacked = _make_unpacked_weight(size_k, size_n, num_bits, device)
    b_q_weight = _gptq_pack(unpacked, num_bits)
    perm = _make_perm(size_k, device)
    expected = _marlin_pack(unpacked[perm.long()], num_bits, False)
    out = torch.empty_like(expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        infini.ops.gptq_marlin_repack(
            b_q_weight,
            perm,
            size_k,
            size_n,
            num_bits,
            False,
            out,
            stream=stream.cuda_stream,
            implementation_index=implementation_index,
        )

    stream.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def _gptq_marlin_repack(
    b_q_weight,
    perm,
    size_k,
    size_n,
    num_bits,
    is_a_8bit,
    out,
    *,
    implementation_index,
):
    infini.ops.gptq_marlin_repack(
        b_q_weight,
        perm,
        size_k,
        size_n,
        num_bits,
        is_a_8bit,
        out,
        stream=get_stream(b_q_weight.device),
        implementation_index=implementation_index,
    )


def _make_unpacked_weight(size_k, size_n, num_bits, device):
    values = torch.arange(size_k * size_n, dtype=torch.int64, device=device)
    values = (values * 17 + 3) % (1 << num_bits)

    return values.reshape(size_k, size_n)


def _make_perm(size_k, device):
    generator = torch.Generator().manual_seed(20260728)

    return torch.randperm(size_k, generator=generator, dtype=torch.int32).to(device)


def _empty_perm(device):
    return torch.empty(0, dtype=torch.int32, device=device)


def _gptq_pack(unpacked, num_bits):
    pack_factor = 32 // num_bits
    packed = torch.zeros(
        (unpacked.size(0) // pack_factor, unpacked.size(1)),
        dtype=torch.int64,
        device=unpacked.device,
    )
    for index in range(pack_factor):
        packed |= unpacked[index::pack_factor] << (num_bits * index)

    return packed.to(torch.int32)


def _marlin_pack(unpacked, num_bits, is_a_8bit):
    size_k, size_n = unpacked.shape
    tile = 16
    if is_a_8bit:
        unpacked = unpacked.reshape(size_k // (tile * 2), tile * 2, -1, tile)
    else:
        unpacked = unpacked.reshape(size_k // tile, tile, -1, tile)

    unpacked = unpacked.permute(0, 2, 1, 3).reshape(size_k // tile, size_n * tile)
    weight_perm = _get_weight_perm(num_bits, is_a_8bit, unpacked.device)
    unpacked = unpacked.reshape(-1, weight_perm.numel())[:, weight_perm]
    unpacked = unpacked.reshape(size_k // tile, size_n * tile)

    pack_factor = 32 // num_bits
    packed = torch.zeros(
        (unpacked.size(0), unpacked.size(1) // pack_factor),
        dtype=torch.int64,
        device=unpacked.device,
    )
    for index in range(pack_factor):
        packed |= unpacked[:, index::pack_factor] << (num_bits * index)

    return packed.to(torch.int32)


def _get_weight_perm(num_bits, is_a_8bit, device):
    perm = []
    if is_a_8bit:
        for index in range(32):
            column = index // 4
            rows = tuple(4 * (index % 4) + offset for offset in range(4))
            rows += tuple(4 * (index % 4 + 4) + offset for offset in range(4))
            block_perm = [
                16 * row + column + 8 * block for block in (0, 1) for row in rows
            ]
            for offset in range(2):
                perm.extend(value + 512 * offset for value in block_perm)
    else:
        for index in range(32):
            column = index // 4
            rows = (
                2 * (index % 4),
                2 * (index % 4) + 1,
                2 * (index % 4 + 4),
                2 * (index % 4 + 4) + 1,
            )
            block_perm = [
                16 * row + column + 8 * block for block in (0, 1) for row in rows
            ]
            for offset in range(4):
                perm.extend(value + 256 * offset for value in block_perm)

    interleave = {
        (4, False): (0, 2, 4, 6, 1, 3, 5, 7),
        (4, True): (0, 4, 1, 5, 2, 6, 3, 7),
        (8, False): (0, 2, 1, 3),
        (8, True): (0, 1, 2, 3),
    }[(num_bits, is_a_8bit)]
    perm_tensor = torch.tensor(perm, dtype=torch.int64)
    perm_tensor = perm_tensor.reshape(-1, len(interleave))[:, interleave]

    return perm_tensor.reshape(-1).to(device)
