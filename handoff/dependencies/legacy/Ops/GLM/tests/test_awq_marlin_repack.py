# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Reference permutations are adapted from vLLM at commit
# `25ace8fe5df07fc13f4aef5a89db391f326e60ee`:
# `vllm/model_executor/layers/quantization/utils/marlin_utils_test.py`.

import infini.ops
import pytest
import torch

from tests.utils import get_stream


if not hasattr(infini.ops, "AwqMarlinRepack"):
    pytest.skip(
        "`AwqMarlinRepack` is not available on this platform",
        allow_module_level=True,
    )


@pytest.mark.parametrize("num_bits", (4, 8))
@pytest.mark.parametrize("is_a_8bit", (False, True))
@pytest.mark.parametrize(("size_k", "size_n"), ((128, 64), (256, 128), (128, 832)))
def test_awq_marlin_repack(
    num_bits,
    is_a_8bit,
    size_k,
    size_n,
    device,
    implementation_index,
):
    if device != "cuda":
        pytest.skip("`awq_marlin_repack` requires the NVIDIA backend")

    unpacked = _make_unpacked_weight(size_k, size_n, num_bits, device)
    b_q_weight = _awq_pack(unpacked, num_bits)
    expected = _marlin_pack(unpacked, num_bits, is_a_8bit)
    out = torch.empty_like(expected)

    _awq_marlin_repack(
        b_q_weight,
        size_k,
        size_n,
        num_bits,
        is_a_8bit,
        out,
        implementation_index=implementation_index,
    )

    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_awq_marlin_repack_non_default_stream(device, implementation_index):
    if device != "cuda":
        pytest.skip("non-default CUDA streams require the NVIDIA backend")

    size_k, size_n, num_bits = 128, 64, 4
    unpacked = _make_unpacked_weight(size_k, size_n, num_bits, device)
    b_q_weight = _awq_pack(unpacked, num_bits)
    expected = _marlin_pack(unpacked, num_bits, False)
    out = torch.empty_like(expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream):
        infini.ops.awq_marlin_repack(
            b_q_weight,
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


def _awq_marlin_repack(
    b_q_weight,
    size_k,
    size_n,
    num_bits,
    is_a_8bit,
    out,
    *,
    implementation_index,
):
    infini.ops.awq_marlin_repack(
        b_q_weight,
        size_k,
        size_n,
        num_bits,
        is_a_8bit,
        out,
        stream=get_stream(b_q_weight.device),
        implementation_index=implementation_index,
    )


def _make_unpacked_weight(size_k, size_n, num_bits, device):
    generator = torch.Generator().manual_seed(size_k * 1009 + size_n * 17 + num_bits)
    values = torch.randint(
        0,
        1 << num_bits,
        (size_k, size_n),
        dtype=torch.int64,
        generator=generator,
    )

    return values.to(device)


def _awq_pack(unpacked, num_bits):
    pack_factor = 32 // num_bits
    undo_pack = {
        4: (0, 4, 1, 5, 2, 6, 3, 7),
        8: (0, 2, 1, 3),
    }[num_bits]
    packed = torch.zeros(
        (unpacked.size(0), unpacked.size(1) // pack_factor),
        dtype=torch.int64,
        device=unpacked.device,
    )
    for logical_position, packed_position in enumerate(undo_pack):
        packed |= unpacked[:, logical_position::pack_factor] << (
            num_bits * packed_position
        )

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
