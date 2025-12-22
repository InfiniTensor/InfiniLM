import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from llava import LLaVAForCauslLM
from libinfinicore_infer import DeviceType


STAGE_NAME_TO_ID = {
    "pre_ln": LLaVAForCauslLM.LLAVA_VISION_STAGE_PRE_LN,
    "select_all": LLaVAForCauslLM.LLAVA_VISION_STAGE_SELECT_ALL,
    "select_patch": LLaVAForCauslLM.LLAVA_VISION_STAGE_SELECT_PATCH,
    "projector": LLaVAForCauslLM.LLAVA_VISION_STAGE_PROJECTOR,
    "projector_all": LLaVAForCauslLM.LLAVA_VISION_STAGE_PROJECTOR_ALL,
}


def _tensor_stats(x: torch.Tensor):
    xf = x.detach().float()
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "abs_max": float(xf.abs().max().item()),
        "abs_mean": float(xf.abs().mean().item()),
        "nan_cnt": int(torch.isnan(xf).sum().item()),
        "inf_cnt": int(torch.isinf(xf).sum().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="Describe this image.")
    ap.add_argument(
        "--stages",
        default="pre_ln,select_patch,projector",
        help="Comma-separated: pre_ln,select_patch,projector",
    )
    ap.add_argument("--out-dir", default="llava_dump_out")
    ap.add_argument("--ndev", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((model_dir / "config.json").read_text())
    image_token_index = int(cfg.get("image_token_index", 32000))

    processor = AutoProcessor.from_pretrained(str(model_dir))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": args.image},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    mm_inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    pixel_values = mm_inputs.pixel_values
    input_ids = mm_inputs.input_ids

    # Print <image> token positions in the raw input_ids (may be a single token in LLaVA v1.5).
    ids = input_ids[0].to(dtype=torch.int64)
    pos = (ids == image_token_index).nonzero(as_tuple=False).flatten().tolist()
    print("input_ids_len:", int(ids.numel()))
    print("image_token_index:", image_token_index)
    print("image_token_positions:", pos)
    torch.save(
        {
            "input_ids": input_ids.cpu(),
            "pixel_values": pixel_values.cpu(),
            "image_token_index": image_token_index,
        },
        out_dir / "inputs.pt",
    )

    img = Image.open(args.image).convert("RGB")
    _ = img  # keep file readable for callers; preprocessing already done by processor

    model = LLaVAForCauslLM(str(model_dir), device=DeviceType.DEVICE_TYPE_HYGON, ndev=args.ndev)

    for name in [s.strip() for s in args.stages.split(",") if s.strip()]:
        if name not in STAGE_NAME_TO_ID:
            raise SystemExit(f"Unknown stage: {name} (valid: {sorted(STAGE_NAME_TO_ID)})")
        stage = STAGE_NAME_TO_ID[name]
        out = model.batch_infer_vision_stage(pixel_values, stage)
        torch.save(out.cpu(), out_dir / f"{name}.pt")
        print(f"{name}:", _tensor_stats(out))
        if args.verbose:
            flat = out.flatten()
            print(f"{name} first10:", [float(x) for x in flat[:10].tolist()])


if __name__ == "__main__":
    main()
