"""Compare fixed real-document prompts with the memory paths enabled or disabled.

Cases are an external JSON list of prompt, token_ids, answers, and max_tokens.
No dataset or checkpoint is downloaded. vLLM is an optional independent backend.
"""

import argparse
import ctypes
import hashlib
import importlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("infinilm", "vllm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--control", action="store_true")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    model_config = json.loads((Path(args.model) / "config.json").read_text())
    if model_config.get("model_type") != "qwen3" or args.tp not in (1, 2):
        raise ValueError("This check requires ordinary Qwen3 with TP1 or TP2.")
    kv_heads = model_config["num_key_value_heads"]
    if kv_heads % args.tp:
        raise ValueError("The reference KV budget requires divisible KV heads.")
    head_dim = model_config.get(
        "head_dim", model_config["hidden_size"] // model_config["num_attention_heads"]
    )
    kv_bytes_per_token = (
        2 * model_config["num_hidden_layers"] * (kv_heads // args.tp) * head_dim * 2
    )
    maximum = max(row["token_count"] + row["max_tokens"] for row in cases)
    result = {
        "status": "RUNNING",
        "framework": args.framework,
        "control": args.control,
        "model": args.model,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "cases": [],
    }
    scores = {}
    capture = []
    native_base = None
    original_forward = None
    model = None

    def save():
        args.result.write_text(json.dumps(result, indent=2))

    save()
    try:
        if args.framework == "infinilm":
            from infinilm import _core as core
            from infinilm.llm.llm import LLM
            from infinilm.llm.sampling_params import SamplingParams

            module = importlib.import_module("infinilm.infer_engine")
            read_config = module.read_hf_config

            def configured(path):
                config = read_config(path)
                config["infinilm_prefill_chunk_size"] = 0 if args.control else 4096
                config["infinilm_tp_vocab_sharding"] = not args.control
                return config

            with patch.object(module, "read_hf_config", configured):
                model = LLM(
                    args.model,
                    device="cuda",
                    dtype="bfloat16",
                    tensor_parallel_size=args.tp,
                    cache_type="paged",
                    attn_backend="paged-attn",
                    max_batch_size=1,
                    static_cache_capacity=1,
                    num_blocks=(maximum + 255) // 256,
                    block_size=256,
                    max_cache_len=maximum,
                    max_tokens=max(row["max_tokens"] for row in cases),
                    enable_graph=False,
                    enable_prefix_caching=False,
                    weight_load_mode="sync",
                )
            engine = model.engine.model_runner.model_engine
            native_base = type(engine).__mro__[1]
            original_forward = native_base.forward

            def traced(self, inputs):
                output = original_forward(self, inputs)
                if not capture:
                    tensor = core.Tensor(output.logits).to("cpu").contiguous()
                    if tensor.dtype == core.bfloat16:
                        bits = np.empty(tensor.shape, np.uint16)
                        ctypes.memmove(bits.ctypes.data, tensor.data_ptr(), bits.nbytes)
                        values = (bits.astype(np.uint32) << 16).view(np.float32)
                    else:
                        values = tensor.to_numpy().copy()
                    capture.append(values.reshape(-1))
                return output

            native_base.forward = traced
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        else:
            import vllm
            from vllm import LLM, SamplingParams

            if args.control:
                raise ValueError("The control switch applies only to InfiniLM.")
            result["vllm_version"] = vllm.__version__
            model = LLM(
                model=args.model,
                dtype="bfloat16",
                tensor_parallel_size=args.tp,
                max_model_len=maximum,
                max_num_seqs=1,
                max_num_batched_tokens=4096,
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                enforce_eager=True,
                kv_cache_dtype="auto",
                kv_cache_memory_bytes=((maximum + 255) // 256 * 256 + 16)
                * kv_bytes_per_token,
                block_size=16,
                swap_space=0,
                cpu_offload_gb=0,
                seed=123,
            )
            tokenizer = model.get_tokenizer()
        for index, case in enumerate(cases):
            assert tokenizer.encode(case["prompt"]) == case["token_ids"]
            capture.clear()
            start = time.perf_counter()
            if args.framework == "infinilm":
                params = SamplingParams(
                    top_k=1,
                    top_p=1.0,
                    temperature=1.0,
                    max_tokens=case["max_tokens"],
                    seed=123,
                )
                outputs = model.generate(
                    prompts=[case["prompt"]], sampling_params=params, use_tqdm=False
                )
            else:
                params = SamplingParams(
                    temperature=0,
                    max_tokens=case["max_tokens"],
                    seed=123,
                    logprobs=20,
                )
                outputs = model.generate(
                    [{"prompt_token_ids": case["token_ids"]}],
                    sampling_params=params,
                    use_tqdm=False,
                )
            output = outputs[0].outputs[0]
            tokens = list(output.token_ids)
            row = {
                key: case[key]
                for key in ("dataset", "index", "id", "token_count", "answers")
            }
            row.update(
                token_ids=tokens,
                text=tokenizer.decode(tokens, skip_special_tokens=True),
                seconds=time.perf_counter() - start,
            )
            if args.framework == "infinilm":
                assert len(capture) == 1
                scores[str(index)] = capture[0]
                np.savez_compressed(args.result.with_suffix(".npz"), **scores)
            else:
                row["first_token_logprobs"] = {
                    str(token): value.logprob
                    for token, value in output.logprobs[0].items()
                }
            result["cases"].append(row)
            save()
            print(
                json.dumps({k: row[k] for k in ("dataset", "index", "seconds")}),
                flush=True,
            )
        if scores:
            np.savez_compressed(args.result.with_suffix(".npz"), **scores)
        result["status"] = "COMPLETE"
    except BaseException as error:
        result.update(status="FAILED", error=repr(error))
        raise
    finally:
        if native_base is not None:
            native_base.forward = original_forward
        if model is not None:
            if args.framework == "infinilm":
                model.close()
            else:
                model.llm_engine.engine_core.shutdown()
        save()


if __name__ == "__main__":
    main()
