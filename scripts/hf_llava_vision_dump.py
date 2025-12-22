import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


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
    ap.add_argument("--out-dir", default="hf_llava_dump_out")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--device", default="cpu", help="e.g. cpu or cuda:0")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((model_dir / "config.json").read_text())
    image_token_index = int(cfg.get("image_token_index", 32000))
    vision_feature_layer = int(cfg.get("vision_feature_layer", -2))

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
            "vision_feature_layer": vision_feature_layer,
        },
        out_dir / "inputs.pt",
    )

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device(args.device)

    model = LlavaForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to(device)
    model.eval()

    pixel_values = pixel_values.to(device=device, dtype=dtype)

    with torch.no_grad():
        vision_model = model.vision_tower.vision_model
        # Node A: embeddings + pre_layrnorm output
        embeds = vision_model.embeddings(pixel_values)
        pre_ln = vision_model.pre_layrnorm(embeds)
        torch.save(pre_ln.cpu(), out_dir / "pre_ln.pt")
        print("pre_ln:", _tensor_stats(pre_ln))

        # Node C: select hidden state from CLIPVisionModel forward (matches LLaVA selection logic)
        vision_out = model.vision_tower(
            pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = vision_out.hidden_states[vision_feature_layer]
        select_patch = hs[:, 1:, :].contiguous()
        torch.save(select_patch.cpu(), out_dir / "select_patch.pt")
        print("select_patch:", _tensor_stats(select_patch))

        projector = model.multi_modal_projector
        proj = projector(select_patch)
        torch.save(proj.cpu(), out_dir / "projector.pt")
        print("projector:", _tensor_stats(proj))

        if args.verbose:
            print("pre_ln first10:", [float(x) for x in pre_ln.flatten()[:10].tolist()])
            print("select_patch first10:", [float(x) for x in select_patch.flatten()[:10].tolist()])
            print("projector first10:", [float(x) for x in proj.flatten()[:10].tolist()])


if __name__ == "__main__":
    main()
