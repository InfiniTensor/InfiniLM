import argparse
from pathlib import Path

import torch


def _stats(x: torch.Tensor):
    xf = x.detach().float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "abs_max": float(xf.abs().max().item()) if xf.numel() else 0.0,
        "abs_mean": float(xf.abs().mean().item()) if xf.numel() else 0.0,
        "nan_cnt": int(torch.isnan(xf).sum().item()),
        "inf_cnt": int(torch.isinf(xf).sum().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--sample-rows", type=int, default=0, help="If >0, compare only first N rows (dim0).")
    args = ap.parse_args()

    a = torch.load(Path(args.a), map_location="cpu")
    b = torch.load(Path(args.b), map_location="cpu")
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise SystemExit("Both inputs must be saved torch.Tensors")

    # Common dump format mismatch: HF dumps include a batch dim [1, ...], while
    # InfiniLM dumps are often saved without it. Normalize by squeezing batch=1.
    if a.ndim >= 3 and a.shape[0] == 1:
        a = a[0].contiguous()
    if b.ndim >= 3 and b.shape[0] == 1:
        b = b[0].contiguous()

    if args.sample_rows > 0 and a.ndim >= 1 and b.ndim >= 1:
        a = a[: args.sample_rows].contiguous()
        b = b[: args.sample_rows].contiguous()

    print("name:", args.name)
    print("a:", args.a, _stats(a))
    print("b:", args.b, _stats(b))

    if a.shape != b.shape:
        print("shape_mismatch:", list(a.shape), list(b.shape))
        return

    af = a.detach().float()
    bf = b.detach().float()
    diff = af - bf
    print(
        "diff:",
        {
            "abs_max": float(diff.abs().max().item()),
            "abs_mean": float(diff.abs().mean().item()),
            "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        },
    )

    # Cosine similarity per row (if 2D)
    if af.ndim == 2 and af.shape[1] > 0:
        an = af / (af.norm(dim=1, keepdim=True) + 1e-12)
        bn = bf / (bf.norm(dim=1, keepdim=True) + 1e-12)
        cos = (an * bn).sum(dim=1)
        print("cos:", {"min": float(cos.min().item()), "mean": float(cos.mean().item()), "max": float(cos.max().item())})


if __name__ == "__main__":
    main()
