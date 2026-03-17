import sys
import os
import time
import re
import csv
import argparse
import json
import numpy as np
from datasets import load_dataset, Dataset
from abc import ABC, abstractmethod


TOTAL_TOKENS = 0
TOTAL_TIME = 0.0


class BaseBenchmark(ABC):
    """Base class for benchmark evaluation with common tokenizer and generation utilities"""

    def encode_text(self, text):
        """Encode text to token IDs - reused across backends"""
        return self.tokenizer.encode(text)

    def decode_token(self, token_id):
        """Decode token ID to text - reused across backends"""
        return self.tokenizer.decode(token_id)

    @abstractmethod
    def render_input_content(self, *args, **kwargs):
        """Render input content - benchmark-specific implementation"""
        pass

    @abstractmethod
    def generate(self, *args, **kwargs):
        """Generate response - benchmark-specific implementation"""
        pass

    @abstractmethod
    def _generate_step(self, tokens, max_steps, topp_, topk_, temperature_):
        """Backend-specific generation implementation"""
        pass


class InfiniLMBenchmark(BaseBenchmark):
    """Wrapper class for InfiniLM cpp backend for benchmark evaluation"""

    def __init__(
        self,
        model_dir_path,
        device_type_str="cpu",
        ndev=1,
        backend="cpp",
        benchmark="ceval",
        enable_paged_attn=False,
        enable_graph=False,
        attn_backend="default",
    ):
        import transformers
        import infinicore
        from infinilm.modeling_utils import load_model_state_dict_by_file
        from infinilm.distributed import DistConfig
        from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig
        from infinilm.infer_engine import InferEngine

        self.benchmark = benchmark

        # Map device type string to infinicore device
        # Note: These map to the Python device type strings used by infinicore.device()
        # which correspond to _TORCH_DEVICE_MAP values in InfiniCore/python/infinicore/device.py
        device_map = {
            "cpu": "cpu",
            "nvidia": "cuda",
            "cambricon": "mlu",
            "ascend": "npu",
            "metax": "cuda",
            "moore": "musa",
            "iluvatar": "cuda",
            "kunlun": "cuda",
            "hygon": "cuda",
            "ali": "cuda",
        }

        device_name = device_map.get(device_type_str.lower(), "cpu")
        # CUDA_VISIBLE_DEVICES is automatically respected by CUDA runtime API
        # When CUDA_VISIBLE_DEVICES=5 is set, CUDA only sees device 5 as device 0
        # So device index 0 will automatically map to the first visible device
        self.device = infinicore.device(device_name, 0)

        # Load config and tokenizer
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            self.config_dict = json.load(f)

        # Align tokenizer initialization with jiuge backend (010)
        # Match the exact same initialization logic based on model type
        model_type = self.config_dict.get("model_type", "")
        if model_type == "llama":
            # For llama models: no trust_remote_code (matches jiuge line 465)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
        elif model_type in ["fm9g", "minicpm", "fm9g7b"]:
            # For fm9g/minicpm/fm9g7b models: use trust_remote_code=True (matches jiuge lines 493-495, 518-520)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
        elif model_type in ["qwen2", "qwen3"]:
            # For qwen2/qwen3 models: no trust_remote_code (matches jiuge line 534-536)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
        else:
            # Default: use trust_remote_code=True for other models
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )

        eos_token_id = self.config_dict.get("eos_token_id")
        self.eos_token_id = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )

        if backend != "cpp":
            raise ValueError(f"Unsupported backend: {backend}.")

        # Create model with cpp backend
        print("Loading model with cpp backend...")
        print(f"Graph compilation: {'enabled' if enable_graph else 'disabled'}")
        print(f"Attention backend: {attn_backend}")

        self.model = InferEngine(
            model_dir_path,
            device=self.device,
            distributed_config=DistConfig(ndev),
            cache_config=(
                PagedKVCacheConfig(128) if enable_paged_attn else StaticKVCacheConfig()
            ),
            enable_graph_compiling=enable_graph,
            attention_backend=attn_backend,
        )

        # Enable KV cache for generation
        self.model.use_cache = True

        # Load weights
        print("Loading model weights...")
        load_model_state_dict_by_file(
            self.model,
            model_dir_path,
            dtype=self.model.config.dtype,
        )
        print("Model loaded successfully")

    def max_context_len(self):
        return self.config_dict.get("max_position_embeddings", 2048)

    def render_input_content(self, *args, **kwargs):
        """Render input content based on benchmark type"""
        if self.benchmark == "ceval":
            return render_ceval(self.tokenizer, *args, **kwargs)
        elif self.benchmark == "mmlu":
            return render_mmlu(self.tokenizer, *args, **kwargs)
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

    def generate(self, *args, max_steps=500, topp_=1.0, topk_=1, temperature_=1.0):
        """Generate response based on benchmark type"""
        # Render input content
        input_content = self.render_input_content(*args)
        print(input_content, end="", flush=True)

        # Encode input
        tokens = self.encode_text(input_content)

        # Delegate to backend-specific generation implementation
        output_content = self._generate_step(
            tokens, max_steps, topp_, topk_, temperature_
        )

        return output_content

    def _generate_step(self, tokens, max_steps, topp_, topk_, temperature_):
        """
        InfiniLM cpp backend-specific generation implementation

        NOTE: Validation confirmed input configs are identical between backends.
        The issue was that manual generation loop called InferEngine.generate() which
        doesn't maintain KV cache. Solution: Use model's built-in generate() method
        which properly handles KV cache through GenerationMixin.
        """
        # Convert tokens to infinicore format
        import infinicore
        from infinilm.infer_engine import GenerationConfig

        input_ids_list = [tokens]
        input_ids = infinicore.from_list(input_ids_list)

        start_time = time.perf_counter()

        # For cpp backend, reset cache before generation if use_cache is enabled
        if (
            self.model.use_cache
            and hasattr(self.model, "_model")
            and hasattr(self.model._model, "reset_cache")
        ):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            max_cache_len = max_steps + seq_len
            self.model.reset_cache(
                batch_size=batch_size, initial_capacity=max_cache_len
            )

        # Use model's built-in generate() method
        output_ids = self.model.generate(
            input_ids=input_ids,
            generation_config=GenerationConfig(
                max_new_tokens=max_steps,
                temperature=temperature_,
                top_k=topk_,
                top_p=topp_,
            ),
        )

        end_time = time.perf_counter()

        # ---- post process ----
        generated_ids = np.array([output_id.to_numpy()[0] for output_id in output_ids])
        output_text = self.tokenizer.decode(generated_ids)

        # ---- stats ----
        input_tokens = len(tokens)
        new_tokens = generated_ids.size
        total_tokens = input_tokens + new_tokens

        total_time = end_time - start_time
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        print(output_text)
        print()
        print(f"Total time: {total_time * 1000:.2f} ms")
        print(f"Input tokens: {input_tokens}")
        print(f"New tokens: {new_tokens}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Throughput: {throughput:.2f} tok/s")
        global TOTAL_TOKENS, TOTAL_TIME
        TOTAL_TOKENS += total_tokens
        TOTAL_TIME += total_time

        return output_text

    def destroy_model_instance(self):
        # Cleanup if needed
        del self.model
        print("Model destroyed")


class TorchBenchmark(BaseBenchmark):
    """Torch backend using HuggingFace Transformers"""

    def __init__(self, model_dir_path, device_type_str="cpu", benchmark="ceval"):
        import torch
        import transformers

        self.benchmark = benchmark

        # Device
        if device_type_str == "nvidia":
            self.device = torch.device("cuda")
        elif device_type_str == "cpu":
            self.device = torch.device("cpu")
        elif device_type_str == "cambricon":
            self.device = torch.device("mlu")
        else:
            raise ValueError(
                f"Torch backend unsupported device type: {device_type_str}"
            )

        # Load tokenizer
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            self.config_dict = json.load(f)

        model_type = self.config_dict.get("model_type", "")
        if model_type in ["fm9g", "minicpm", "fm9g7b"]:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )

        # Load model
        print("Loading model with torch backend...")
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_dir_path,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()
        print("Torch model loaded successfully")

        eos_token_id = self.config_dict.get("eos_token_id")
        self.eos_token_id = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )

    def max_context_len(self):
        return self.config_dict.get("max_position_embeddings", 2048)

    def render_input_content(self, *args, **kwargs):
        if self.benchmark == "ceval":
            return render_ceval(self.tokenizer, *args, **kwargs)
        elif self.benchmark == "mmlu":
            return render_mmlu(self.tokenizer, *args, **kwargs)
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

    def _generate_step(self, tokens, max_steps, topp_, topk_, temperature_):
        import torch
        import time

        input_ids = torch.tensor([tokens], device=self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        outputs = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_steps,
            do_sample=temperature_ > 0,
            temperature=temperature_,
            top_k=topk_,
            top_p=topp_,
            eos_token_id=self.eos_token_id,
            pad_token_id=2,
        )

        # --- end sync ---
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # ---- post process ----
        generated_ids = outputs[0][len(tokens) :]
        output_text = self.tokenizer.decode(generated_ids)

        # ---- stats ----
        input_tokens = len(tokens)
        new_tokens = generated_ids.numel()
        total_tokens = input_tokens + new_tokens

        total_time = end_time - start_time
        throughput = total_tokens / total_time if total_time > 0 else 0.0
        print(output_text)
        print()
        print(f"Total time: {total_time * 1000:.2f} ms")
        print(f"Input tokens: {input_tokens}")
        print(f"New tokens: {new_tokens}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Throughput: {throughput:.2f} tok/s")
        global TOTAL_TOKENS, TOTAL_TIME
        TOTAL_TOKENS += total_tokens
        TOTAL_TIME += total_time

        return output_text

    def generate(self, *args, max_steps=500, topp_=1.0, topk_=1, temperature_=1.0):
        input_content = self.render_input_content(*args)
        print(input_content, end="", flush=True)

        tokens = self.encode_text(input_content)

        return self._generate_step(tokens, max_steps, topp_, topk_, temperature_)

    def destroy_model_instance(self):
        del self.model
        print("Torch model destroyed")


class VLLMBenchmark(BaseBenchmark):
    """vLLM backend using vllm.LLM"""

    def __init__(
        self,
        model_dir_path,
        device_type_str="nvidia",
        tensor_parallel_size=1,
        benchmark="ceval",
    ):
        import transformers
        from vllm import LLM

        if device_type_str == "cpu":
            raise ValueError("vLLM backend does not support CPU device type.")

        self.benchmark = benchmark

        # ---- tokenizer ----
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            self.config_dict = json.load(f)

        model_type = self.config_dict.get("model_type", "")
        if model_type in ["qwen2", "qwen3"]:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )

        eos_token_id = self.config_dict.get("eos_token_id")
        self.eos_token_id = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )

        # vLLM engine
        print("Loading model with vLLM backend...")
        self.llm = LLM(
            model=model_dir_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        print("vLLM model loaded successfully")

    def max_context_len(self):
        return self.config_dict.get("max_position_embeddings", 2048)

    def render_input_content(self, *args, **kwargs):
        if self.benchmark == "ceval":
            return render_ceval(self.tokenizer, *args, **kwargs)
        elif self.benchmark == "mmlu":
            return render_mmlu(self.tokenizer, *args, **kwargs)
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

    def generate(self, *args, max_steps=500, topp_=1.0, topk_=1, temperature_=1.0):
        input_content = self.render_input_content(*args)
        print(input_content, end="", flush=True)

        tokens = self.encode_text(input_content)
        return self._generate_step(tokens, max_steps, topp_, topk_, temperature_)

    def _generate_step(self, tokens, max_steps, topp_, topk_, temperature_):
        from vllm import SamplingParams

        prompt = self.tokenizer.decode(tokens)

        sampling_params = SamplingParams(
            max_tokens=max_steps,
            temperature=temperature_,
            top_p=topp_,
            top_k=topk_,
            stop_token_ids=self.eos_token_id,
        )

        start_time = time.perf_counter()

        outputs = self.llm.generate(
            prompts=[prompt],
            sampling_params=sampling_params,
        )

        end_time = time.perf_counter()

        # ---- post process ----
        output_text = outputs[0].outputs[0].text

        # ---- stats ----
        input_tokens = len(tokens)
        new_tokens = len(self.encode_text(output_text))
        total_tokens = input_tokens + new_tokens

        total_time = end_time - start_time
        throughput = total_tokens / total_time if total_time > 0 else 0.0

        print(output_text)
        print()
        print(f"Total time: {total_time * 1000:.2f} ms")
        print(f"Input tokens: {input_tokens}")
        print(f"New tokens: {new_tokens}")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Throughput: {throughput:.2f} tok/s")

        global TOTAL_TOKENS, TOTAL_TIME
        TOTAL_TOKENS += total_tokens
        TOTAL_TIME += total_time

        return output_text

    def destroy_model_instance(self):
        del self.llm
        print("vLLM model destroyed")


def render_ceval(_tokenizer, conversation):
    """Render C-Eval conversation to input content"""
    return (
        _tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        + "正确答案是"
    )


def render_mmlu(_tokenizer, question, choices):
    """Render MMLU question and choices to input content"""
    choices_text = "\n".join(
        [f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)]
    )
    instruction = (
        "You are a multiple-choice question solver. "
        "Select the correct option and respond with only the letter A, B, C, or D."
    )
    prompt = f"{instruction}\n\nQuestion: {question}\n{choices_text}\nAnswer:"

    # Use chat template if available, otherwise return plain text
    if hasattr(_tokenizer, "apply_chat_template"):
        conversation = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"{question}\n{choices_text}\n"},
        ]
        try:
            return (
                _tokenizer.apply_chat_template(
                    conversation=conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
                + "The answer is: "
            )
        except Exception:
            return prompt
    return prompt


def extract_answer_ceval(output_content, answer):
    """Extract predicted answer from C-Eval output"""
    output_upper = output_content.upper().strip()
    position = 0
    ABCD = output_upper[position : position + 2]
    return answer in ABCD


def extract_answer_mmlu(output_content):
    """Extract predicted answer from MMLU output (returns 0-3 index or None)"""
    output_upper = output_content.upper().strip()

    # Find first meaningful token
    match = re.search(r"\b([ABCD])\b", output_upper)
    if match:
        return ord(match.group(1)) - ord("A")
    else:
        match_num = re.search(r"\b([0-3])\b", output_upper)
        if match_num:
            return int(match_num.group(1))
    return None


def evaluate_samples(model, samples, benchmark, max_new_tokens, subject_name=None):
    """Evaluate samples for a single subject and return results"""
    answers_list = []
    for idx, sample in enumerate(samples):
        if benchmark == "ceval":
            input_content = f"'question':{sample['question']},'A': {sample['A']}, 'B':{sample['B']}, 'C': {sample['C']},'D': {sample['D']}。"
            conversation = [
                {
                    "role": "system",
                    "content": "请从question的A，B，C，D四个选项中选择正确的选项。例如，标准答案：A。",
                },
                {"role": "user", "content": input_content},
            ]
            answer = sample["answer"]
            output_content = model.generate(
                conversation,
                max_steps=max_new_tokens,
                topp_=1.0,
                topk_=1,
                temperature_=1.0,
            )
            is_correct = extract_answer_ceval(output_content, answer)
            answers_list.append(
                {
                    "id": sample.get("id", idx),
                    "output_content": output_content,
                    "answer": answer,
                    "is_correct": is_correct,
                    "subject": subject_name,
                }
            )
            if benchmark == "ceval":
                print("标准答案：", answer)

        elif benchmark == "mmlu":
            question = sample["question"]
            choices = sample["choices"]
            answer_idx = sample["answer"]  # MMLU answer is 0-3 index

            output_content = model.generate(
                question,
                choices,
                max_steps=max_new_tokens,
                topp_=1.0,
                topk_=1,
                temperature_=1.0,
            )

            predicted_answer = extract_answer_mmlu(output_content)

            # Convert answer index to letter for display
            answer_letter = chr(65 + answer_idx) if answer_idx < 4 else "?"
            predicted_letter = (
                chr(65 + predicted_answer)
                if predicted_answer is not None and predicted_answer < 4
                else "?"
            )

            print(
                f"Sample {idx}: Correct answer: {answer_letter} ({answer_idx}), Predicted: {predicted_letter} ({predicted_answer})"
            )

            answers_list.append(
                {
                    "id": idx,
                    "output_content": output_content,
                    "answer": answer_idx,
                    "predicted": predicted_answer,
                    "subject": subject_name,
                }
            )

    # Evaluate results for this subject
    true_num = 0
    all_num = 0
    for cont in answers_list:
        id = cont["id"]
        all_num = all_num + 1

        if benchmark == "ceval":
            answer = cont["answer"]
            is_correct = cont["is_correct"]
            if is_correct:
                true_num = true_num + 1
                print(f"id {id} : ", "正确")
            else:
                print(f"id {id}: ", "错误")

        elif benchmark == "mmlu":
            answer = cont["answer"]
            predicted = cont["predicted"]
            if predicted is not None and predicted == answer:
                true_num = true_num + 1
                print(f"id {id}: Correct")
            else:
                answer_letter = chr(65 + answer) if answer < 4 else "?"
                predicted_letter = (
                    chr(65 + predicted)
                    if predicted is not None and predicted < 4
                    else "?"
                )
                print(
                    f"id {id}: Wrong (correct: {answer_letter}, predicted: {predicted_letter})"
                )

    accuracy = true_num / all_num if all_num > 0 else 0.0
    if benchmark == "ceval":
        print(f"成绩: {true_num}/{all_num}", accuracy)
    else:
        print(f"Accuracy: {true_num}/{all_num} = {accuracy:.2%}")

    return {
        "subject": subject_name or "all",
        "correct": true_num,
        "total": all_num,
        "accuracy": accuracy,
        "answers_list": answers_list,
    }


def _load_ceval_from_cache(cache_dir, subject_name, split, ceval_subjects):
    """
    Load CEval data from local cache avoiding network calls.
    Scans cached Arrow files under ceval___ceval-exam and filters by split.
    """
    split_names = (
        ["test"] if split == "test" else ["val"] if split == "val" else ["val", "test"]
    )

    base = os.path.join(cache_dir, "ceval___ceval-exam", subject_name)
    if os.path.isdir(base):
        records = []
        for root, _, files in os.walk(base):
            for fname in files:
                if not fname.endswith(".arrow"):
                    continue
                lower = fname.lower()
                if split == "test" and "test" not in lower:
                    continue
                if split == "val" and not any(
                    x in lower for x in ["val", "validation", "dev"]
                ):
                    continue
                if split == "all" and not any(
                    x in lower for x in ["val", "validation", "dev", "test"]
                ):
                    continue
                try:
                    ds = Dataset.from_file(os.path.join(root, fname))
                    records.extend(ds.to_list())
                except Exception:
                    continue
        if records:
            return records

    # If cache_dir provided and nothing loaded, fail without network
    raise FileNotFoundError(
        f"CEval cached data not found for subject '{subject_name}' with splits {split_names}"
    )


def _load_mmlu_from_cache(cache_dir, subject_name, split, mmlu_subjects):
    """
    Load MMLU data from local cache avoiding network calls.
    Scans cached Arrow files under cache_dir/cais___mmlu and filters by split.
    """

    def load_one(subj):
        split_names = (
            ["test"]
            if split == "test"
            else (
                ["validation", "dev"]
                if split == "val"
                else ["validation", "dev", "test"]
            )
        )

        base = os.path.join(cache_dir, "cais___mmlu", subj)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"MMLU cache dir not found: {base}")

        records = []
        for root, _, files in os.walk(base):
            for fname in files:
                if not fname.endswith(".arrow"):
                    continue
                lower = fname.lower()
                if split == "test" and "test" not in lower:
                    continue
                if split == "val" and not any(
                    x in lower for x in ["validation", "dev"]
                ):
                    continue
                if split == "all" and not any(
                    x in lower for x in ["validation", "dev", "test"]
                ):
                    continue
                try:
                    ds = Dataset.from_file(os.path.join(root, fname))
                    records.extend(ds.to_list())
                except Exception:
                    continue
        if records:
            return records
        raise FileNotFoundError(
            f"MMLU cached data not found for subject '{subj}' with splits {split_names}"
        )

    if subject_name == "all":
        # Use hardcoded list of MMLU subjects, excluding "all"
        all_samples = []
        for subj in mmlu_subjects:
            try:
                all_samples.extend(load_one(subj))
            except FileNotFoundError:
                continue
        if not all_samples:
            raise FileNotFoundError(
                f"No MMLU cached data found for any subject. Please ensure datasets are cached."
            )
        return all_samples, "all"

    return load_one(subject_name), subject_name


def parse_list(value: str):
    """
    Parse list argument: can be a single int or a list of ints.

    Examples:
        "1" -> 1
        "[1,2,4]" -> [1, 2, 4]
        "1,2,4" -> [1, 2, 4]
    """
    value = value.strip()
    # Try to parse as JSON list first
    if value.startswith("[") and value.endswith("]"):
        try:
            result = json.loads(value)
            if isinstance(result, list):
                return [int(x) for x in result]
            return int(result)
        except (json.JSONDecodeError, ValueError):
            pass

    # Try to parse as comma-separated values
    if "," in value:
        try:
            return [int(x.strip()) for x in value.split(",")]
        except ValueError:
            pass

    # Try to parse as a single integer
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Value must be an int or list[int], got: {value}"
        )


def parse_arguments():
    """Parse command line arguments using argparse"""
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for language models on CEval and MMLU datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_benchmark.py --cpu /path/to/model --bench ceval --backend cpp
  python test_benchmark.py --nvidia /path/to/model --bench mmlu --backend vllm --ndev 2
  python test_benchmark.py --cpu /path/to/model --bench ceval --subject "accountant" --num_samples 10 --output_csv results.csv
        """,
    )

    # Device flags (mutually exclusive)
    device_group = parser.add_mutually_exclusive_group(required=True)
    device_group.add_argument("--cpu", action="store_true", help="Use CPU device")
    device_group.add_argument(
        "--nvidia", action="store_true", help="Use NVIDIA GPU device"
    )
    device_group.add_argument(
        "--cambricon", action="store_true", help="Use Cambricon MLU device"
    )
    device_group.add_argument("--ascend", action="store_true", help="Use Ascend device")
    device_group.add_argument("--metax", action="store_true", help="Use Metax device")
    device_group.add_argument("--moore", action="store_true", help="Use Moore device")
    device_group.add_argument(
        "--iluvatar", action="store_true", help="Use Iluvatar device"
    )
    device_group.add_argument("--kunlun", action="store_true", help="Use Kunlun device")
    device_group.add_argument("--hygon", action="store_true", help="Use Hygon device")
    device_group.add_argument("--ali", action="store_true", help="Use Ali device")

    # Positional argument for model path
    parser.add_argument("model_path", type=str, help="Path to the model directory")

    # Required benchmark argument
    parser.add_argument(
        "--bench",
        required=True,
        choices=["ceval", "mmlu"],
        help="Benchmark to evaluate (ceval or mmlu)",
    )

    # Optional arguments
    parser.add_argument(
        "--backend",
        type=str,
        default="cpp",
        choices=["python", "cpp", "torch", "vllm"],
        help="Backend to use for inference (default: cpp)",
    )
    parser.add_argument(
        "--ndev",
        type=int,
        default=1,
        help="Number of devices for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="all",
        help="Subject(s) to evaluate, comma-separated or 'all' (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "all"],
        help="Dataset split to use: test, val, or all (default: test)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate per subject (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=500,
        help="Maximum number of new tokens to generate (default: 500)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Path to output CSV file for results",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to use for dataset cache (offline mode when specified)",
    )

    # InfiniLM specific options
    parser.add_argument(
        "--enable-paged-attn",
        action="store_true",
        help="Enable paged attention for InfiniLM backend",
    )
    parser.add_argument(
        "--enable-graph",
        action="store_true",
        help="Enable graph compilation for InfiniLM backend",
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="default",
        choices=["default", "flash-attn"],
        help="Attention backend for InfiniLM (default: default)",
    )

    return parser.parse_args()


def load_dataset_samples(args):
    """
    Load dataset samples based on benchmark type and subject list.
    Returns a dictionary mapping subject names to their samples.
    """
    # Parse comma-separated subjects
    if args.subject and args.subject != "all":
        subject_list = [s.strip() for s in args.subject.split(",")]
    else:
        subject_list = ["all"]

    # Define helper functions for loading datasets
    if args.bench == "ceval":
        ceval_subjects = [
            "accountant",
            "advanced_mathematics",
            "art_studies",
            "basic_medicine",
            "business_administration",
            "chinese_language_and_literature",
            "civil_servant",
            "clinical_medicine",
            "college_chemistry",
            "college_economics",
            "college_physics",
            "college_programming",
            "computer_architecture",
            "computer_network",
            "discrete_mathematics",
            "education_science",
            "electrical_engineer",
            "environmental_impact_assessment_engineer",
            "fire_engineer",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_chinese",
            "high_school_geography",
            "high_school_history",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_politics",
            "ideological_and_moral_cultivation",
            "law",
            "legal_professional",
            "logic",
            "mao_zedong_thought",
            "marxism",
            "metrology_engineer",
            "middle_school_biology",
            "middle_school_chemistry",
            "middle_school_geography",
            "middle_school_history",
            "middle_school_mathematics",
            "middle_school_physics",
            "middle_school_politics",
            "modern_chinese_history",
            "operating_system",
            "physician",
            "plant_protection",
            "probability_and_statistics",
            "professional_tour_guide",
            "sports_science",
            "tax_accountant",
            "teacher_qualification",
            "urban_and_rural_planner",
            "veterinary_medicine",
        ]

        def _load_ceval_subject(subj):
            print(f"Loading C-Eval dataset (subject: {subj})...")
            if args.cache_dir:
                return _load_ceval_from_cache(
                    args.cache_dir, subj, args.split, ceval_subjects
                )
            # online fallback via HF load_dataset
            if args.split == "all":
                records = []
                for split_name in ["val", "test"]:
                    try:
                        ds = load_dataset(
                            r"ceval/ceval-exam", name=subj, split=split_name
                        )
                        records.extend(ds.to_list())
                    except Exception:
                        continue
                if records:
                    return records
                raise FileNotFoundError(
                    f"No ceval splits found online for subject {subj}"
                )
            hf_split = "test" if args.split == "test" else "val"
            ds = load_dataset(r"ceval/ceval-exam", name=subj, split=hf_split)
            data = ds.to_list()
            return data

        def load_subject_samples(subj_name):
            if subj_name == "all":
                samples = []
                for subj in ceval_subjects:
                    samples.extend(_load_ceval_subject(subj))
                return samples, "all"
            else:
                if subj_name not in ceval_subjects:
                    raise ValueError(
                        f"Unknown C-Eval subject '{subj_name}'. Available subjects: {', '.join(ceval_subjects)}"
                    )
                return _load_ceval_subject(subj_name), subj_name

    elif args.bench == "mmlu":
        mmlu_subjects = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]

        def _load_mmlu_subject(subj):
            print(f"Loading MMLU dataset (subject: {subj})...")
            if args.cache_dir:
                return _load_mmlu_from_cache(
                    args.cache_dir, subj, args.split, mmlu_subjects
                )
            if subj == "all":
                samples = []
                splits_to_load = (
                    ["test"]
                    if args.split == "test"
                    else (
                        ["validation"]
                        if args.split == "val"
                        else ["validation", "test"]
                    )
                )
                # Load each subject individually from hardcoded list, excluding "all"
                for subject_name in mmlu_subjects:
                    for sp in splits_to_load:
                        try:
                            dataset = load_dataset("cais/mmlu", subject_name, split=sp)
                            if hasattr(dataset, "to_list"):
                                samples.extend(dataset.to_list())
                            else:
                                samples.extend(list(dataset))
                        except Exception:
                            continue
                if not samples:
                    raise FileNotFoundError(
                        f"No MMLU data found for any subject in the list"
                    )
                return samples, "all"
            else:
                splits_to_load = (
                    ["test"]
                    if args.split == "test"
                    else (
                        ["validation"]
                        if args.split == "val"
                        else ["validation", "test"]
                    )
                )
                records = []
                for sp in splits_to_load:
                    try:
                        dataset = load_dataset("cais/mmlu", subj, split=sp)
                        if hasattr(dataset, "to_list"):
                            records.extend(dataset.to_list())
                        else:
                            records.extend(list(dataset))
                    except Exception:
                        continue
                if not records:
                    raise FileNotFoundError(
                        f"MMLU subject {subj} split(s) {splits_to_load} not found"
                    )
                return records, subj

        def load_subject_samples(subj_name):
            return _load_mmlu_subject(subj_name)

    # Load samples for each subject
    subject_samples = {}

    # Expand "all" to individual subjects for per-subject reporting
    if "all" in subject_list:
        if args.bench == "ceval":
            # Replace "all" with all individual ceval subjects
            expanded_subjects = [s for s in subject_list if s != "all"] + ceval_subjects
        elif args.bench == "mmlu":
            # Replace "all" with all individual mmlu subjects
            expanded_subjects = [s for s in subject_list if s != "all"] + mmlu_subjects
    else:
        expanded_subjects = subject_list

    # Remove duplicates while preserving order
    expanded_subjects = list(dict.fromkeys(expanded_subjects))

    for subj in expanded_subjects:
        print(f"\n{'=' * 60}")
        print(f"Loading dataset for subject: {subj}")
        print(f"{'=' * 60}\n")

        try:
            samples, actual_subj_name = load_subject_samples(subj)
            print(f"Loaded {len(samples)} samples for subject: {actual_subj_name}")

            # Limit number of samples if specified
            if args.num_samples is not None and args.num_samples > 0:
                original_count = len(samples)
                samples = samples[: args.num_samples]
                print(
                    f"Limited to {len(samples)} samples for evaluation (from {original_count} total)"
                )

            if len(samples) > 0:
                subject_samples[actual_subj_name] = samples
            else:
                print(f"No samples found for subject: {actual_subj_name}")

        except Exception as e:
            print(f"Error loading subject '{subj}': {e}")
            continue

    return subject_samples


def main():
    """Main function"""
    args = parse_arguments()

    # Map device flags to device type string
    device_type_str = "cpu"
    if args.cpu:
        device_type_str = "cpu"
    elif args.nvidia:
        device_type_str = "nvidia"
    elif args.cambricon:
        device_type_str = "cambricon"
    elif args.ascend:
        device_type_str = "ascend"
    elif args.metax:
        device_type_str = "metax"
    elif args.moore:
        device_type_str = "moore"
    elif args.iluvatar:
        device_type_str = "iluvatar"
    elif args.kunlun:
        device_type_str = "kunlun"
    elif args.hygon:
        device_type_str = "hygon"
    elif args.ali:
        device_type_str = "ali"

    # Normalize cache_dir and force offline when provided
    if args.cache_dir:
        args.cache_dir = os.path.expanduser(args.cache_dir)
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Step 1: Load dataset samples first
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60 + "\n")

    subject_samples = load_dataset_samples(args)

    if not subject_samples:
        print("No samples loaded. Exiting.")
        return

    # Step 2: Create model based on backend
    print("\n" + "=" * 60)
    print("STEP 2: LOADING MODEL")
    print("=" * 60 + "\n")

    if args.backend == "torch":
        assert args.ndev == 1, "Torch backend only supports single-device evaluation"
        model = TorchBenchmark(args.model_path, device_type_str, args.bench)
    elif args.backend == "vllm":
        model = VLLMBenchmark(args.model_path, device_type_str, args.ndev, args.bench)
    else:  # cpp backend
        model = InfiniLMBenchmark(
            args.model_path,
            device_type_str,
            args.ndev,
            args.backend,
            args.bench,
            args.enable_paged_attn,
            args.enable_graph,
            args.attn,
        )

    # Step 3: Evaluate each subject
    print("\n" + "=" * 60)
    print("STEP 3: EVALUATING")
    print("=" * 60 + "\n")

    all_results = []

    for subject_name, samples in subject_samples.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating subject: {subject_name}")
        print(f"{'=' * 60}\n")

        # Evaluate samples for this subject
        result = evaluate_samples(
            model, samples, args.bench, args.max_new_tokens, subject_name
        )
        all_results.append(result)
        print(
            f"\nSubject '{subject_name}' completed: {result['correct']}/{result['total']} = {result['accuracy']:.2%}"
        )

    model.destroy_model_instance()

    # Calculate overall results
    print(f"\n{'=' * 60}")
    print("OVERALL RESULTS")
    print(f"{'=' * 60}")
    if len(all_results) == 0:
        print("No tests were run.")
        return
    elif len(all_results) > 1:
        for r in all_results:
            print(
                f"Subject '{r['subject']}': {r['correct']}/{r['total']} = {r['accuracy']:.2%}"
            )
    overall_correct = sum(r["correct"] for r in all_results)
    overall_total = sum(r["total"] for r in all_results)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    print(f"{'=' * 60}")
    if args.bench == "ceval":
        print(
            f"Overall 成绩: {overall_correct}/{overall_total} = {overall_accuracy:.2%}"
        )
    else:
        print(
            f"Overall Accuracy: {overall_correct}/{overall_total} = {overall_accuracy:.2%}"
        )

    print(f"Total Latency: {TOTAL_TIME:.2f} seconds")
    print(f"Total Tokens Processed: {TOTAL_TOKENS} tokens")
    if TOTAL_TIME > 0:
        print(f"Overall Throughput: {TOTAL_TOKENS / TOTAL_TIME:.2f} tokens/s")

    # Write CSV if output path is specified
    if args.output_csv:
        print(f"\nWriting results to CSV: {args.output_csv}")
        with open(args.output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Subject", "Correct", "Total", "Accuracy"])
            for result in all_results:
                writer.writerow(
                    [
                        result["subject"],
                        result["correct"],
                        result["total"],
                        f"{result['accuracy']:.4f}",
                    ]
                )
            writer.writerow(
                ["Overall", overall_correct, overall_total, f"{overall_accuracy:.4f}"]
            )
        print(f"CSV file written successfully: {args.output_csv}")


if __name__ == "__main__":
    main()
