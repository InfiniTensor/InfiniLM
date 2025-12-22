import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="Describe this image.")
    ap.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=1, help="Greedy generate steps including the first token.")
    ap.add_argument("--use-fast", action="store_true", help="Use fast image processor if available")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    image_token_index = int(cfg.get("image_token_index", 32000))

    processor = AutoProcessor.from_pretrained(str(model_dir), use_fast=args.use_fast)
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

    input_ids = mm_inputs.input_ids
    pixel_values = mm_inputs.pixel_values
    attention_mask = getattr(mm_inputs, "attention_mask", None)

    ids = input_ids[0].to(dtype=torch.int64)
    pos = (ids == image_token_index).nonzero(as_tuple=False).flatten().tolist()
    print("input_ids_len:", int(ids.numel()))
    print("image_token_index:", image_token_index)
    print("image_token_count:", len(pos))
    if len(pos) > 0:
        print("image_token_pos_first_last:", (pos[0], pos[-1]))

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    model = LlavaForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    ).to(device)
    model.eval()

    input_ids = input_ids.to(device)
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = out.logits[0, -1].float().cpu()
        next_id = int(torch.argmax(logits).item())
        print("next_token_id:", next_id)
        topk = min(args.topk, int(logits.numel()))
        vals, idx = torch.topk(logits, k=topk)
        pairs = [(int(i), float(v)) for i, v in zip(idx.tolist(), vals.tolist())]
        print("last_logits_topk:", pairs)

        if int(args.max_new_tokens) > 1:
            gen = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=int(args.max_new_tokens),
                use_cache=True,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=model.config.eos_token_id,
            )
            new_tokens = gen[0, input_ids.shape[1] :].tolist()
            print("generated_token_ids:", new_tokens)


if __name__ == "__main__":
    main()
