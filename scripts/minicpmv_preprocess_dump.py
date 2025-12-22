import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image


def _load_preprocessor_config(model_dir: Path) -> dict:
    cand = model_dir / "preprocessor_config.json"
    if cand.exists():
        return json.loads(cand.read_text())
    # fallback to the vendored config
    here = Path(__file__).resolve().parent.parent
    return json.loads((here / "minicpmv_config" / "preprocessor_config.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to an input image")
    ap.add_argument("--max-slice-nums", type=int, default=None)
    args = ap.parse_args()

    model_dir = Path(os.environ.get("MINICPMV_MODEL_DIR", "")).expanduser()
    if not str(model_dir):
        model_dir = Path(".")

    cfg = _load_preprocessor_config(model_dir)
    max_slice_nums = cfg.get("max_slice_nums", 9) if args.max_slice_nums is None else args.max_slice_nums

    from minicpmv_config.image_processing_minicpmv import MiniCPMVImageProcessor

    ip = MiniCPMVImageProcessor(
        max_slice_nums=cfg.get("max_slice_nums", 9),
        scale_resolution=cfg.get("scale_resolution", 448),
        patch_size=cfg.get("patch_size", 14),
        use_image_id=cfg.get("use_image_id", True),
        image_feature_size=cfg.get("image_feature_size", 64),
        im_start=cfg.get("im_start", "<image>"),
        im_end=cfg.get("im_end", "</image>"),
        slice_start=cfg.get("slice_start", "<slice>"),
        slice_end=cfg.get("slice_end", "</slice>"),
        unk=cfg.get("unk", "<unk>"),
        im_id_start=cfg.get("im_id_start", "<image_id>"),
        im_id_end=cfg.get("im_id_end", "</image_id>"),
        slice_mode=cfg.get("slice_mode", True),
        norm_mean=cfg.get("norm_mean", [0.5, 0.5, 0.5]),
        norm_std=cfg.get("norm_std", [0.5, 0.5, 0.5]),
        version=cfg.get("version", 2.6),
    )

    img = Image.open(args.image).convert("RGB")
    out = ip.preprocess(img, do_pad=True, max_slice_nums=max_slice_nums, return_tensors="pt")

    # out["pixel_values"] is a nested list: [batch][slice] where each slice is [3, patch, L]
    pv = out["pixel_values"]
    tgt = out["tgt_sizes"]
    sizes = out["image_sizes"]

    print("preprocess dump:")
    print("  image_size:", sizes[0][0] if sizes and sizes[0] else None)
    print("  num_slices:", len(pv[0]) if pv and pv[0] else 0)
    for i, t in enumerate(tgt[0]):
        th, tw = int(t[0].item()), int(t[1].item())
        seq_len = th * tw
        x = pv[0][i]
        print(f"  slice[{i}]: tgt_h={th} tgt_w={tw} seq_len={seq_len} pixel_values={tuple(x.shape)} dtype={x.dtype}")
        # Expected packed layout for our C++ APIs: [1, 3, patch, seq_len*patch]
        packed = x.unsqueeze(0)
        ok = (packed.ndim == 4) and (packed.shape[0] == 1) and (packed.shape[1] == 3) and (packed.shape[2] == cfg.get("patch_size", 14))
        ok = ok and (packed.shape[3] == seq_len * cfg.get("patch_size", 14))
        print("    packed_ok:", bool(ok))

    # Simple checksum for determinism debugging
    if pv and pv[0]:
        v0 = pv[0][0].float()
        print("  first_slice_stats:", float(v0.mean().item()), float(v0.abs().mean().item()), float(v0.abs().max().item()))


if __name__ == "__main__":
    main()

