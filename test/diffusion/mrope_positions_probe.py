"""Compare the InfiniOps `Mrope` kernel with the checkpoint's rotary for each positions layout.

usage: python test/diffusion/mrope_positions_probe.py [CASE ...]

`Mrope` accepts packed ``[tokens]`` positions or ``[3, tokens]`` axis rows.
"""

import sys

import infini.ops as ops
import torch

HEAD, ROTARY, SECTIONS = 256, 64, (16, 8, 8)
QUERY_HEADS, KEY_HEADS, TOKENS, TABLE = 16, 2, 40, 2048
CASES = ("1d", "3d_same", "3d_distinct")


def reference(value, cos, sin, axis_rows):
    """Rotate like the checkpoint's `CloverAttention.rotary`, one position per axis."""
    width = ROTARY // 2
    axis_of_pair = torch.cat(
        [
            torch.full((section,), index, device=value.device)
            for index, section in enumerate(SECTIONS)
        ]
    )
    position = axis_rows[axis_of_pair].T
    pair = torch.arange(width, device=value.device)[None, :]
    phase_cos, phase_sin = (
        cos[position, pair][:, None, :],
        sin[position, pair][:, None, :],
    )
    view = value.view(TOKENS, -1, HEAD)
    x, y = view[:, :, :width].float(), view[:, :, width:ROTARY].float()
    out = value.clone().view(TOKENS, -1, HEAD)
    out[:, :, :width] = x * phase_cos - y * phase_sin
    out[:, :, width:ROTARY] = y * phase_cos + x * phase_sin
    return out.reshape(value.shape)


def case_inputs(case, device):
    """Return the positions tensor and the axis rows whose results are compared."""
    rows = torch.arange(TOKENS, device=device, dtype=torch.int64)
    packed = ("packed positions", rows.expand(3, -1))
    if case == "1d":
        return rows, [packed]
    if case == "3d_same":
        return rows.expand(3, -1).contiguous(), [packed]
    if case == "3d_distinct":
        distinct = torch.stack((rows, rows + 5, rows + 100))
        return distinct, [("three distinct axis rows", distinct)]
    raise ValueError(f"Unknown layout {case!r}.")


def main():
    cases = sys.argv[1:] or list(CASES)
    if set(cases) - set(CASES):
        raise SystemExit(f"Unknown case. Choose from {', '.join(CASES)}.")
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)
    query = torch.randn(TOKENS, QUERY_HEADS * HEAD, device=device).to(dtype)
    key = torch.randn(TOKENS, KEY_HEADS * HEAD, device=device).to(dtype)
    # A random table makes every pair sensitive to the position it reads.
    cos = torch.randn(TABLE, ROTARY // 2, device=device).to(dtype)
    sin = torch.randn(TABLE, ROTARY // 2, device=device).to(dtype)
    stream = torch.cuda.current_stream().cuda_stream
    for case in cases:
        positions, expectations = case_inputs(case, device)
        query_out, key_out = torch.empty_like(query), torch.empty_like(key)
        print(f"case={case} positions shape={tuple(positions.shape)}", flush=True)
        ops.mrope(
            query,
            key,
            cos,
            sin,
            positions,
            HEAD,
            ROTARY,
            *SECTIONS,
            False,
            query_out,
            key_out,
            stream=stream,
        )
        torch.cuda.synchronize()
        for label, axis_rows in expectations:
            for value, out in ((query, query_out), (key, key_out)):
                want = reference(value, cos, sin, axis_rows).to(out.dtype)
                torch.testing.assert_close(out, want, rtol=0.02, atol=0.03)
            print(f"  matches {label}: query and key")


if __name__ == "__main__":
    main()
