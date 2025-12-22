import argparse
import json
import os
import time
from ctypes import POINTER, byref, c_float, c_int, c_uint
from pathlib import Path

import safetensors
import torch
from transformers import AutoProcessor

from libinfinicore_infer import DeviceType, JiugeModel, KVCacheCStruct

# Reuse helpers from scripts/llava.py (LLaVA naming for language_model.* weights + Jiuge packer)
from llava import JiugeMetaFromLlama, JiugeWeightsImpl, LlamaWeightsNaming, LLaVAForCauslLM


def load_all_safetensors_from_dir(dir_path: str):
    tensors = {}
    dir_path = Path(dir_path)
    for file in sorted(dir_path.glob("*.safetensors")):
        data = safetensors.safe_open(file, "pt")
        for name in data.keys():
            tensors[name] = data.get_tensor(name)
    return tensors


def expand_image_tokens(input_ids: list[int], image_token_index: int, n_image_tokens: int):
    out_tokens: list[int] = []
    override_pos: list[int] = []
    for token in input_ids:
        if token == image_token_index:
            start = len(out_tokens)
            out_tokens.extend([image_token_index] * n_image_tokens)
            override_pos.extend(list(range(start, start + n_image_tokens)))
        else:
            out_tokens.append(token)
    return out_tokens, override_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="Describe this image.")
    ap.add_argument("--ndev", type=int, default=1)
    ap.add_argument("--device", default="hygon", choices=["hygon"])
    ap.add_argument(
        "--image-tokens",
        type=int,
        default=576,
        help="How many tokens <image> expands to for overrides (try 576 first; try 577 if HF merge uses an extra token).",
    )
    ap.add_argument("--dump-topk", type=int, default=10, help="Top-k logits to print (requires dumping logits to host).")
    ap.add_argument("--dump-logits", action="store_true", help="Also dump last-token logits to host (slow).")
    ap.add_argument("--max-new-tokens", type=int, default=1, help="Greedy decode steps including the first token.")
    ap.add_argument("--decode", action="store_true", help="Decode generated tokens with the tokenizer.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_dir = Path(args.model_dir)
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
    input_ids = mm_inputs.input_ids[0].tolist()
    image_pos = [i for i, t in enumerate(input_ids) if t == image_token_index]
    print("raw_input_len:", len(input_ids))
    print("image_token_index:", image_token_index)
    print("image_token_positions:", image_pos)
    print("image_token_count:", len(image_pos))

    # Stage 1: get image embeds from projector (patch-only by default)
    llava = LLaVAForCauslLM(str(model_dir), device=DeviceType.DEVICE_TYPE_HYGON, ndev=args.ndev)
    if args.image_tokens == 576:
        img_embeds = llava.batch_infer_vision_stage(
            pixel_values, LLaVAForCauslLM.LLAVA_VISION_STAGE_PROJECTOR
        )
    elif args.image_tokens == 577:
        img_embeds = llava.batch_infer_vision_stage(
            pixel_values, LLaVAForCauslLM.LLAVA_VISION_STAGE_PROJECTOR_ALL
        )
    else:
        raise SystemExit("--image-tokens must be 576 or 577 for now")
    img_embeds = img_embeds.contiguous()
    print("img_embeds:", tuple(img_embeds.shape), img_embeds.dtype)

    # Stage 2: build overrides.
    # LLaVA v1.5 processors often already expand to 576 <image> tokens in input_ids.
    # If not, expand a single <image> token into N tokens.
    if len(image_pos) == int(img_embeds.shape[0]):
        expanded = input_ids
        override_pos_list = image_pos
    elif len(image_pos) == 1 and int(img_embeds.shape[0]) == int(args.image_tokens):
        expanded, override_pos_list = expand_image_tokens(input_ids, image_token_index, args.image_tokens)
    else:
        raise SystemExit(
            "Cannot match image token positions to image embeds. "
            f"image_token_count={len(image_pos)} embed_rows={int(img_embeds.shape[0])} "
            f"(try --image-tokens 576 or 577)."
        )

    print("expanded_len:", len(expanded))
    print("override_cnt:", len(override_pos_list))
    override_pos = (c_uint * len(override_pos_list))(*override_pos_list)

    # Stage 3: run Jiuge prefill with overrides.
    print("Loading language model weights...")
    t0 = time.time()
    state_dict = load_all_safetensors_from_dir(str(model_dir))
    meta = JiugeMetaFromLlama(cfg.get("text_config", cfg), dtype=torch.float16, max_tokens=cfg.get("max_position_embeddings", 4096))
    naming = LlamaWeightsNaming()
    weights = JiugeWeightsImpl(meta, naming, state_dict, ndev=args.ndev)
    jiuge = JiugeModel()
    dev_ids = (c_int * args.ndev)(*[i for i in range(args.ndev)])
    model = jiuge.create_model(byref(meta), byref(weights), DeviceType.DEVICE_TYPE_HYGON, args.ndev, dev_ids)
    print(f"LM load seconds: {time.time() - t0:.3f}")

    ntok = len(expanded)
    tokens_c = (c_uint * ntok)(*expanded)
    req_lens = (c_uint * 1)(ntok)
    req_pos = (c_uint * 1)(0)

    kv = jiuge.create_kv_cache(
        meta.nlayer,
        meta.dctx,
        meta.nkvh,
        meta.dh,
        meta.dh,
        meta.dt_logits,
        DeviceType.DEVICE_TYPE_HYGON,
        dev_ids,
        args.ndev,
    )
    kv_caches = (POINTER(KVCacheCStruct) * 1)(kv)

    # Greedy
    temperature = (c_float * 1)(0.0)
    topk = (c_uint * 1)(1)
    topp = (c_float * 1)(1.0)
    out = (c_uint * 1)()

    jiuge.infer_batch_with_overrides(
        model,
        tokens_c,
        ntok,
        req_lens,
        1,
        req_pos,
        kv_caches,
        len(override_pos_list),
        override_pos,
        img_embeds.data_ptr(),
        temperature,
        topk,
        topp,
        out,
    )
    next_id = int(out[0])
    print("next_token_id:", next_id)
    generated = [next_id]

    rope_pos = ntok
    kv_pos = ntok
    eos_id = 2

    for _ in range(max(0, int(args.max_new_tokens) - 1)):
        if generated[-1] == eos_id:
            break
        req_lens1 = (c_uint * 1)(1)
        req_pos1 = (c_uint * 1)(rope_pos)
        tokens1 = (c_uint * 1)(generated[-1])
        jiuge.infer_batch(
            model,
            tokens1,
            1,
            req_lens1,
            1,
            req_pos1,
            kv_caches,
            temperature,
            topk,
            topp,
            out,
        )
        generated.append(int(out[0]))
        rope_pos += 1
        kv_pos += 1

    print("generated_token_ids:", generated)
    if args.decode:
        tok = processor.tokenizer
        print("decoded:", tok.decode(generated, skip_special_tokens=False))

    if args.dump_logits:
        # Dump full logits for all input tokens (slow, large), then report only the last position topk.
        dvoc = int(meta.dvoc)
        logits = torch.empty((ntok, dvoc), dtype=torch.float16, device="cpu")
        jiuge.forward_batch_with_overrides(
            model,
            tokens_c,
            ntok,
            req_lens,
            1,
            req_pos,
            kv_caches,
            len(override_pos_list),
            override_pos,
            img_embeds.data_ptr(),
            logits.data_ptr(),
        )
        last = logits[-1].float()
        topk_vals, topk_idx = torch.topk(last, k=min(args.dump_topk, dvoc))
        pairs = [(int(i), float(v)) for i, v in zip(topk_idx.tolist(), topk_vals.tolist())]
        print("last_logits_topk:", pairs)

    jiuge.drop_kv_cache(kv)
    jiuge.destroy_model(model)


if __name__ == "__main__":
    main()
