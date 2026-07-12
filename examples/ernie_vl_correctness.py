import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time

import infinicore
import numpy as np
import torch
from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file
from transformers import AutoProcessor, AutoTokenizer

EXPECTED_IDS = {
    "text": [
        3843,
        5971,
        94036,
        31282,
        5502,
        965,
        93956,
        5119,
        94111,
        6385,
        5188,
        1555,
        94035,
        8217,
        94110,
        586,
    ],
    "image": [38020, 432, 93938, 1981, 93968, 93927, 1505, 93937],
    "video": [3843, 1510, 1386, 94001, 5434, 1187, 6350, 93956],
}


def build_video():
    video = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    video[0, :, :, 0] = 40
    video[0, :, :, 1] = 120
    video[0, :, :, 2] = 200
    video[1, :, :, 0] = 120
    video[1, :, :, 1] = 120
    video[1, :, :, 2] = 140
    return video


def apply_template(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_inputs(case, model_path, processor, tokenizer, image_path):
    if case == "text":
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "用一句话介绍你自己。"}],
            }
        ]
        text = apply_template(tokenizer, messages)
        inputs = processor(text=[text], return_tensors="pt")
        return text, inputs

    if case == "image":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_path,
                            "image_width": 224,
                            "image_height": 224,
                        },
                    },
                ],
            }
        ]
        text = apply_template(tokenizer, messages)
        image_inputs, video_inputs = processor.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return text, inputs

    if case == "video":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "dummy_numpy_2x32x32x3"},
                    {"type": "text", "text": "Describe the video briefly."},
                ],
            }
        ]
        text = apply_template(tokenizer, messages)
        inputs = processor(text=text, videos=[build_video()], return_tensors="pt")
        return text, inputs

    raise ValueError(f"unknown case: {case}")


def infini_from_torch(tensor):
    return infinicore.from_torch(tensor.contiguous())


def build_hf_references(args, model_path, processor, tokenizer, image_path):
    child_code = r"""
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def build_video():
    video = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    video[0, :, :, 0] = 40
    video[0, :, :, 1] = 120
    video[0, :, :, 2] = 200
    video[1, :, :, 0] = 120
    video[1, :, :, 1] = 120
    video[1, :, :, 2] = 140
    return video


def apply_template(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_inputs(case, processor, tokenizer, image_path):
    if case == "text":
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "用一句话介绍你自己。"}],
            }
        ]
        text = apply_template(tokenizer, messages)
        inputs = processor(text=[text], return_tensors="pt")
        return text, inputs

    if case == "image":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_path,
                            "image_width": 224,
                            "image_height": 224,
                        },
                    },
                ],
            }
        ]
        text = apply_template(tokenizer, messages)
        image_inputs, video_inputs = processor.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return text, inputs

    if case == "video":
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "dummy_numpy_2x32x32x3"},
                    {"type": "text", "text": "Describe the video briefly."},
                ],
            }
        ]
        text = apply_template(tokenizer, messages)
        inputs = processor(text=text, videos=[build_video()], return_tensors="pt")
        return text, inputs

    raise ValueError(f"unknown case: {case}")


def torch_dtype_from_name(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def move_inputs_to_device(inputs, device):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def fix_hf_meta_helper_tensors(model):
    patched = []
    vision_model = getattr(model, "vision_model", None) or getattr(model, "visual", None)
    rotary = getattr(vision_model, "rotary_pos_emb", None) if vision_model is not None else None
    inv_freq = getattr(rotary, "inv_freq", None) if rotary is not None else None
    if inv_freq is not None and getattr(inv_freq, "device", None).type == "meta":
        dim = int(inv_freq.numel() * 2)
        theta = float(getattr(rotary, "theta", 10000.0))
        rotary.inv_freq = 1.0 / theta ** (
            torch.arange(start=0, end=dim, step=2, dtype=torch.float32) / dim
        )
        patched.append("vision_rotary_inv_freq")

    for module in model.modules():
        experts_type_ids = getattr(module, "experts_type_ids", None)
        if experts_type_ids is None or getattr(experts_type_ids, "device", None).type != "meta":
            continue
        config = getattr(module, "config", None)
        moe_num_experts = getattr(config, "moe_num_experts", None)
        if not isinstance(moe_num_experts, (list, tuple)):
            continue
        rebuilt = torch.zeros([sum(moe_num_experts)], dtype=torch.int64)
        offset = 0
        masks = []
        for idx, expert_num in enumerate(moe_num_experts):
            rebuilt[offset : offset + expert_num] = idx
            offset += expert_num
        module.experts_type_ids = rebuilt
        for idx, _ in enumerate(moe_num_experts):
            masks.append(module.experts_type_ids == idx)
        module.experts_type_mask = masks
        patched.append("experts_type_ids")
    return patched


config_path, output_path = sys.argv[1], sys.argv[2]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

processor = AutoProcessor.from_pretrained(
    cfg["model_path"],
    trust_remote_code=True,
    video_min_pixels=3136,
    video_max_pixels=3136,
)
tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg["model_path"],
    device_map=cfg["hf_device_map"],
    torch_dtype=torch_dtype_from_name(cfg["hf_torch_dtype"]),
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    use_flash_attention=cfg["hf_use_flash_attention"],
)
model.eval()
patched_meta = fix_hf_meta_helper_tensors(model)
if patched_meta:
    print(f"patched HF meta helper tensors: {patched_meta}", flush=True)
if hasattr(model, "add_image_preprocess"):
    model.add_image_preprocess(processor)

references = {}
details = []
for case in cfg["cases"]:
    text, inputs = build_inputs(case, processor, tokenizer, cfg["image_path"])
    prompt_len = int(inputs["input_ids"].shape[-1])
    model_inputs = move_inputs_to_device(inputs, model.device)
    max_new_tokens = cfg["hf_max_new_tokens_by_case"][case]
    start = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            inputs=model_inputs["input_ids"],
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=cfg["hf_use_cache"],
        )
    elapsed = time.time() - start
    token_ids = generated_ids[0][prompt_len:].detach().cpu().tolist()
    item = {
        "hf_reference": case,
        "prompt_len": prompt_len,
        "new_token_ids": token_ids,
        "output_text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "elapsed_sec": elapsed,
    }
    references[case] = token_ids
    details.append(item)
    print(
        json.dumps(
            {
                **item,
                "elapsed_sec": round(elapsed, 3),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"references": references, "details": details}, f, ensure_ascii=False)
"""

    config = {
        "model_path": model_path,
        "image_path": image_path,
        "cases": args.cases,
        "hf_device_map": args.hf_device_map,
        "hf_torch_dtype": args.hf_torch_dtype,
        "hf_use_cache": args.hf_use_cache,
        "hf_use_flash_attention": args.hf_use_flash_attention,
        "hf_max_new_tokens_by_case": {
            case: args.hf_max_new_tokens or len(EXPECTED_IDS[case])
            for case in args.cases
        },
    }

    with tempfile.TemporaryDirectory(prefix="ernie_hf_reference_") as tmp_dir:
        script_path = os.path.join(tmp_dir, "run_hf_reference.py")
        config_path = os.path.join(tmp_dir, "config.json")
        output_path = os.path.join(tmp_dir, "reference.json")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(child_code)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)

        subprocess.run(
            [sys.executable, script_path, config_path, output_path], check=True
        )
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.load(f)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result["references"]


def summarize_inputs(inputs):
    summary = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            item = {"shape": list(value.shape), "dtype": str(value.dtype)}
            if key in {"grid_thw", "image_type_ids"}:
                item["value"] = value.detach().cpu().tolist()
            summary[key] = item
        else:
            summary[key] = str(type(value))
    return summary


def parse_tp_devices(value):
    device_ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            device_ids.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid device id in --tp-devices: {item!r}"
            ) from exc

    if not device_ids:
        raise argparse.ArgumentTypeError("--tp-devices must contain at least one id")
    if any(device_id < 0 for device_id in device_ids):
        raise argparse.ArgumentTypeError("--tp-devices cannot contain negative ids")
    return device_ids


def build_dist_config(args):
    if args.tp < 1:
        raise ValueError("--tp must be >= 1")

    if args.tp_devices is not None:
        tp_device_ids = args.tp_devices
        if args.tp != 1 and args.tp != len(tp_device_ids):
            raise ValueError(
                f"--tp ({args.tp}) must match --tp-devices length ({len(tp_device_ids)})"
            )
        dist_config = DistConfig(tp_device_ids=tp_device_ids)
    else:
        tp_device_ids = list(range(args.tp))
        dist_config = DistConfig(args.tp)

    if args.device == "cuda":
        device_count = torch.cuda.device_count()
        if device_count < len(tp_device_ids):
            raise ValueError(
                f"tensor parallel needs {len(tp_device_ids)} CUDA device(s), "
                f"but torch sees {device_count}"
            )
        invalid_ids = [
            device_id for device_id in tp_device_ids if device_id >= device_count
        ]
        if invalid_ids:
            raise ValueError(
                f"--tp-devices contains unavailable CUDA device id(s): {invalid_ids}; "
                f"torch sees device ids 0..{device_count - 1}"
            )

    return dist_config, tp_device_ids


def run_case(engine, processor, tokenizer, case, text, inputs, expected_ids):
    max_new_tokens = len(expected_ids)
    kwargs = {}
    if inputs.get("position_ids") is not None:
        kwargs["position_ids"] = infini_from_torch(
            inputs["position_ids"].to(torch.int64)
        )
    if inputs.get("token_type_ids") is not None:
        kwargs["token_type_ids"] = infini_from_torch(
            inputs["token_type_ids"].to(torch.int64)
        )
    if inputs.get("images") is not None:
        kwargs["images"] = infini_from_torch(inputs["images"].contiguous())
        kwargs["grid_thw"] = infini_from_torch(inputs["grid_thw"].to(torch.int64))
        kwargs["image_type_ids"] = infini_from_torch(
            inputs["image_type_ids"].to(torch.int64)
        )

    input_ids = infini_from_torch(inputs["input_ids"].to(torch.int64))
    start = time.time()
    output = engine.generate(
        input_ids,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=1,
            top_p=1.0,
        ),
        **kwargs,
    )
    elapsed = time.time() - start

    token_ids = []
    for tensor in output:
        token_ids.extend(np.array(tensor.to_numpy()).reshape(-1).astype(int).tolist())

    return {
        "case": case,
        "prompt": text,
        "prompt_len": int(inputs["input_ids"].shape[-1]),
        "input_summary": summarize_inputs(inputs),
        "expected_token_ids": expected_ids,
        "new_token_ids": token_ids,
        "match_expected": token_ids == expected_ids,
        "output_text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "elapsed_sec": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--tp-devices", type=parse_tp_devices, default=None)
    parser.add_argument(
        "--reference-mode",
        choices=["expected", "hf"],
        default="expected",
        help="Use baked HF token baselines or run the HF model live first.",
    )
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument(
        "--hf-torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--hf-max-new-tokens", type=int, default=None)
    parser.add_argument("--hf-use-cache", action="store_true")
    parser.add_argument("--hf-use-flash-attention", action="store_true")
    parser.add_argument("--cases", nargs="+", default=["text", "image", "video"])
    parser.add_argument("--image", default=None)
    parser.add_argument("--max-cache-len", type=int, default=512)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = os.path.expanduser(args.model)
    image_path = args.image or os.path.join(model_path, "benchmark.jpg")
    dist_config, tp_device_ids = build_dist_config(args)
    run_config = {
        "device": args.device,
        "tp_device_ids": tp_device_ids,
        "dist_config": str(dist_config),
        "max_cache_len": args.max_cache_len,
        "reference_mode": args.reference_mode,
    }
    print(json.dumps({"run_config": run_config}, ensure_ascii=False), flush=True)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        video_min_pixels=3136,
        video_max_pixels=3136,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    for case in args.cases:
        if case not in EXPECTED_IDS:
            raise ValueError(f"unsupported case: {case}")

    if args.reference_mode == "hf":
        expected_ids_by_case = build_hf_references(
            args, model_path, processor, tokenizer, image_path
        )
    else:
        expected_ids_by_case = {case: EXPECTED_IDS[case] for case in args.cases}

    engine = InferEngine(
        model_path,
        device=infinicore.device(args.device, 0),
        distributed_config=dist_config,
        cache_config=StaticKVCacheConfig(
            max_batch_size=1, max_cache_len=args.max_cache_len
        ),
        attention_backend="default",
    )
    load_model_state_dict_by_file(engine, model_path, dtype=engine.dtype)

    results = []
    for case in args.cases:
        text, inputs = build_inputs(case, model_path, processor, tokenizer, image_path)
        engine.reset_cache(
            StaticKVCacheConfig(max_batch_size=1, max_cache_len=args.max_cache_len)
        )
        result = run_case(
            engine,
            processor,
            tokenizer,
            case,
            text,
            inputs,
            expected_ids_by_case[case],
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "case": result["case"],
                    "prompt_len": result["prompt_len"],
                    "new_token_ids": result["new_token_ids"],
                    "match_expected": result["match_expected"],
                    "output_text": result["output_text"],
                    "elapsed_sec": round(result["elapsed_sec"], 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    ok = all(item["match_expected"] for item in results)
    report = {"ok": ok, "run_config": run_config, "results": results}
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
