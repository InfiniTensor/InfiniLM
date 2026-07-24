import argparse
import ast
import csv
import json
import os
import re
import tempfile
import time

import numpy as np
import torch
from datasets import get_dataset_config_names, load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["hf", "cpp"], default="hf")
    parser.add_argument("--subjects", default="Accounting")
    parser.add_argument("--split", choices=["dev", "validation"], default="validation")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--prompt-style",
        choices=["official", "direct", "ernie"],
        default="official",
        help="official follows MMMU's default prompt format; direct/ernie are experimental.",
    )
    parser.add_argument("--max-cache-len", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--tp-devices", type=parse_tp_devices, default=None)
    parser.add_argument(
        "--torch-dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def parse_tp_devices(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def torch_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def selected_subjects(subjects_arg):
    if subjects_arg == "all":
        return get_dataset_config_names("MMMU/MMMU")
    return [item.strip() for item in subjects_arg.split(",") if item.strip()]


def parse_options(raw):
    if isinstance(raw, list):
        return raw
    try:
        value = ast.literal_eval(raw)
    except Exception:
        value = raw
    if isinstance(value, list):
        return [str(item) for item in value]
    return [part.strip() for part in str(value).split("\n") if part.strip()]


def normalize_question(question):
    text = question
    for idx in range(1, 8):
        text = text.replace(f"<image {idx}>", "")
    return " ".join(text.split())


def extract_answer(text, choices=("A", "B", "C", "D"), index2ans=None):
    output_upper = text.upper().strip()
    answer_matches = re.findall(r"ANSWER\s*[:：]\s*([ABCD])\b", output_upper)
    if answer_matches:
        return answer_matches[-1]
    final_matches = re.findall(r"FINAL\s+ANSWER\s*[:：]\s*([ABCD])\b", output_upper)
    if final_matches:
        return final_matches[-1]
    response = " " + output_upper.strip(",.!?;:'") + " "
    candidates = []
    positions = []
    for choice in choices:
        for pattern in (f"({choice})", f" {choice} "):
            pos = response.rfind(pattern)
            if pos >= 0:
                candidates.append(choice)
                positions.append(pos)
    if not candidates and index2ans and len(response.split()) > 5:
        for choice, answer_text in index2ans.items():
            pos = response.lower().rfind(str(answer_text).lower())
            if pos >= 0:
                candidates.append(choice)
                positions.append(pos)
    if candidates:
        return candidates[int(np.argmax(positions))]
    return ""


def extract_debug_answers(text, choices=("A", "B", "C", "D")):
    output_upper = text.upper()
    choice_class = "".join(choices)
    boxed_matches = re.findall(
        rf"(?:BOXED|\\BOXED)\s*\{{\s*([{choice_class}])\s*\}}", output_upper
    )
    explicit_matches = re.findall(
        rf"(?:ANSWER\s*(?:IS|:|：)|FINAL\s+ANSWER\s*(?:IS|:|：))\s*\(?([{choice_class}])\)?",
        output_upper,
    )
    return {
        "boxed_answer": boxed_matches[-1] if boxed_matches else "",
        "explicit_answer": explicit_matches[-1] if explicit_matches else "",
    }


def build_messages(row, image_paths, image_size, prompt_style):
    options = parse_options(row["options"])
    question = normalize_question(row["question"])
    if prompt_style == "official":
        choices_text = "".join(
            f"({chr(65 + idx)}) {choice}\n" for idx, choice in enumerate(options)
        )
        text = (
            f"{question} {choices_text}"
            "Answer with the option's letter from the given choices directly."
        )
    elif prompt_style == "ernie":
        choices_text = "\n".join(
            f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(options)
        )
        text = (
            "Please solve this MMMU multiple-choice problem using the provided image(s). "
            "You may reason internally, but the final response must contain the answer "
            "letter in the form 'Answer: A', 'Answer: B', 'Answer: C', or 'Answer: D'.\n\n"
            f"Question: {question}\n{choices_text}"
        )
    else:
        choices_text = "\n".join(
            f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(options)
        )
        text = (
            "Answer the multiple-choice question. Respond with only the letter "
            "A, B, C, or D.\n\n"
            f"Question: {question}\n{choices_text}"
        )
    content = []
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": path,
                    "image_width": image_size,
                    "image_height": image_size,
                },
            }
        )
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def save_images(row, tmp_dir):
    paths = []
    for idx in range(1, 8):
        image = row.get(f"image_{idx}")
        if image is None:
            continue
        path = os.path.join(tmp_dir, f"image_{idx}.png")
        image.save(path)
        paths.append(path)
    return paths


def build_inputs(processor, tokenizer, row, tmp_dir, image_size, prompt_style):
    image_paths = save_images(row, tmp_dir)
    messages = build_messages(row, image_paths, image_size, prompt_style)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = processor.process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return prompt, inputs, len(image_paths)


def tensor_to_infini(tensor):
    import infinicore

    return infinicore.from_torch(tensor.contiguous())


def fix_hf_meta_helper_tensors(model):
    patched = []
    vision_model = getattr(model, "vision_model", None) or getattr(
        model, "visual", None
    )
    rotary = (
        getattr(vision_model, "rotary_pos_emb", None)
        if vision_model is not None
        else None
    )
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
        if (
            experts_type_ids is None
            or getattr(experts_type_ids, "device", None).type != "meta"
        ):
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


class HFRunner:
    def __init__(self, model_path, device, dtype_name):
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype(dtype_name),
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        patched_meta = fix_hf_meta_helper_tensors(self.model)
        if patched_meta:
            print(f"patched HF meta helper tensors: {patched_meta}", flush=True)
        if hasattr(self.model, "add_image_preprocess"):
            self.model.add_image_preprocess(self.processor)

    def generate(self, row, max_new_tokens, tmp_dir, image_size, prompt_style):
        prompt, inputs, image_count = build_inputs(
            self.processor, self.tokenizer, row, tmp_dir, image_size, prompt_style
        )
        model_inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
            if value is not None
        }
        if "attention_mask" not in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
        prompt_len = int(model_inputs["input_ids"].shape[-1])
        start = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
            )
        elapsed = time.time() - start
        ids = outputs[0][prompt_len:].detach().cpu().tolist()
        return {
            "prompt_len": prompt_len,
            "image_count": image_count,
            "output_ids": ids,
            "output_text": self.tokenizer.decode(ids, skip_special_tokens=True),
            "elapsed_sec": elapsed,
        }


class CppRunner:
    def __init__(self, model_path, device, max_cache_len, tp=1, tp_devices=None):
        import infinicore
        from infinilm.cache import StaticKVCacheConfig
        from infinilm.distributed import DistConfig
        from infinilm.infer_engine import InferEngine
        from infinilm.modeling_utils import load_model_state_dict_by_file

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.max_cache_len = max_cache_len
        if tp_devices is not None and len(tp_devices) != tp:
            raise ValueError(
                f"--tp-devices length {len(tp_devices)} does not match --tp {tp}"
            )
        dist_config = (
            DistConfig(tp_device_ids=tp_devices)
            if tp_devices is not None
            else DistConfig(tp)
        )
        self.engine = InferEngine(
            model_path,
            device=infinicore.device(device, 0),
            distributed_config=dist_config,
            cache_config=StaticKVCacheConfig(
                max_batch_size=1, max_cache_len=max_cache_len
            ),
            attention_backend="default",
        )
        load_model_state_dict_by_file(self.engine, model_path, dtype=self.engine.dtype)

    def generate(self, row, max_new_tokens, tmp_dir, image_size, prompt_style):
        from infinilm.cache import StaticKVCacheConfig
        from infinilm.infer_engine import GenerationConfig

        prompt, inputs, image_count = build_inputs(
            self.processor, self.tokenizer, row, tmp_dir, image_size, prompt_style
        )
        kwargs = {}
        if inputs.get("position_ids") is not None:
            kwargs["position_ids"] = tensor_to_infini(
                inputs["position_ids"].to(torch.int64)
            )
        if inputs.get("token_type_ids") is not None:
            kwargs["token_type_ids"] = tensor_to_infini(
                inputs["token_type_ids"].to(torch.int64)
            )
        if inputs.get("images") is not None:
            kwargs["images"] = tensor_to_infini(inputs["images"].contiguous())
            kwargs["grid_thw"] = tensor_to_infini(inputs["grid_thw"].to(torch.int64))
            kwargs["image_type_ids"] = tensor_to_infini(
                inputs["image_type_ids"].to(torch.int64)
            )

        input_ids = tensor_to_infini(inputs["input_ids"].to(torch.int64))
        self.engine.reset_cache(
            StaticKVCacheConfig(max_batch_size=1, max_cache_len=self.max_cache_len)
        )
        start = time.time()
        output = self.engine.generate(
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
        ids = []
        for tensor in output:
            ids.extend(np.array(tensor.to_numpy()).reshape(-1).astype(int).tolist())
        return {
            "prompt_len": int(inputs["input_ids"].shape[-1]),
            "image_count": image_count,
            "output_ids": ids,
            "output_text": self.tokenizer.decode(ids, skip_special_tokens=True),
            "elapsed_sec": elapsed,
        }


def main():
    args = parse_args()
    if args.backend == "hf":
        if args.tp != 1 or args.tp_devices is not None:
            raise ValueError(
                "--tp and --tp-devices only configure the InfiniLM cpp backend; "
                "HF multi-GPU reference uses transformers device_map via --device auto."
            )
        if args.device == "cuda":
            print(
                "warning: HF backend with --device cuda loads on one GPU; "
                "use --device auto for multi-GPU device_map reference runs.",
                flush=True,
            )
        runner = HFRunner(args.model, args.device, args.torch_dtype)
    else:
        runner = CppRunner(
            args.model,
            args.device,
            args.max_cache_len,
            tp=args.tp,
            tp_devices=args.tp_devices,
        )

    rows = []
    official_outputs = {}
    total_correct = 0
    total_count = 0
    total_skipped = 0

    for subject in selected_subjects(args.subjects):
        dataset = load_dataset("MMMU/MMMU", subject, split=args.split)
        subject_correct = 0
        subject_count = 0
        with tempfile.TemporaryDirectory(prefix=f"mmmu_{subject}_") as tmp_dir:
            for row_index, row in enumerate(dataset):
                if args.num_samples is not None and subject_count >= args.num_samples:
                    break
                if row.get("question_type") != "multiple-choice":
                    total_skipped += 1
                    continue
                result = runner.generate(
                    row,
                    args.max_new_tokens,
                    tmp_dir,
                    args.image_size,
                    args.prompt_style,
                )
                options = parse_options(row["options"])
                choices = [chr(65 + idx) for idx in range(len(options))]
                index2ans = {choice: options[idx] for idx, choice in enumerate(choices)}
                pred = extract_answer(result["output_text"], choices, index2ans)
                debug_answers = extract_debug_answers(result["output_text"], choices)
                gold = str(row["answer"]).strip().upper()[:1]
                ok = pred == gold
                subject_correct += int(ok)
                total_correct += int(ok)
                subject_count += 1
                total_count += 1
                item = {
                    "subject": subject,
                    "id": row.get("id", row_index),
                    "gold": gold,
                    "pred": pred,
                    "ok": int(ok),
                    "prompt_len": result["prompt_len"],
                    "image_count": result["image_count"],
                    "elapsed_sec": f"{result['elapsed_sec']:.3f}",
                    "new_tokens": len(result["output_ids"]),
                    "hit_max_new_tokens": int(
                        len(result["output_ids"]) >= args.max_new_tokens
                    ),
                    **debug_answers,
                    "output_ids": " ".join(str(x) for x in result["output_ids"]),
                    "output_text": result["output_text"].replace("\n", "\\n"),
                }
                rows.append(item)
                official_outputs[str(row.get("id", row_index))] = (
                    pred or result["output_text"]
                )
                print(json.dumps(item, ensure_ascii=False), flush=True)

        acc = subject_correct / subject_count if subject_count else 0.0
        print(
            f"SUBJECT {subject} {subject_correct}/{subject_count} {acc:.4f}", flush=True
        )

    overall = total_correct / total_count if total_count else 0.0
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "subject",
            "id",
            "gold",
            "pred",
            "ok",
            "prompt_len",
            "image_count",
            "elapsed_sec",
            "new_tokens",
            "hit_max_new_tokens",
            "boxed_answer",
            "explicit_answer",
            "output_ids",
            "output_text",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "backend": args.backend,
        "subjects": selected_subjects(args.subjects),
        "split": args.split,
        "prompt_style": args.prompt_style,
        "image_size": args.image_size,
        "tp": args.tp if args.backend == "cpp" else None,
        "tp_devices": args.tp_devices if args.backend == "cpp" else None,
        "correct": total_correct,
        "total": total_count,
        "skipped": total_skipped,
        "accuracy": overall,
        "csv": args.output_csv,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    **report,
                    "official_eval_only_outputs": official_outputs,
                    "rows": rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
