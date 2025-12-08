import sys
import os
import argparse
import time
import re
from datasets import load_dataset
import infinicore
import infinilm
from infinilm.models.llama import AutoLlamaModel
from infinilm.modeling_utils import get_model_state_dict
from infinilm.distributed import DistConfig
from abc import ABC, abstractmethod


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

    def __init__(self, model_dir_path, device_type_str="cpu", ndev=1, backend="cpp", benchmark="ceval"):
        import transformers

        self.benchmark = benchmark

        # Map device type string to infinicore device
        device_map = {
            "cpu": "cpu",
            "nvidia": "cuda",
            "cambricon": "cambricon",
            "ascend": "ascend",
            "metax": "metax",
            "moore": "moore",
            "iluvatar": "iluvatar",
            "kunlun": "kunlun",
            "hygon": "hygon",
        }

        device_name = device_map.get(device_type_str.lower(), "cpu")
        # CUDA_VISIBLE_DEVICES is automatically respected by CUDA runtime API
        # When CUDA_VISIBLE_DEVICES=5 is set, CUDA only sees device 5 as device 0
        # So device index 0 will automatically map to the first visible device
        self.device = infinicore.device(device_name, 0)
        self.dtype = infinicore.bfloat16

        # Load config and tokenizer
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            import json
            self.config_dict = json.load(f)

        # Align tokenizer initialization with jiuge backend (010)
        # Match the exact same initialization logic based on model type
        model_type = self.config_dict.get("model_type", "")
        if model_type == "llama":
            # For llama models: no trust_remote_code (matches jiuge line 465)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)
        elif model_type in ["fm9g", "minicpm", "fm9g7b"]:
            # For fm9g/minicpm/fm9g7b models: use trust_remote_code=True (matches jiuge lines 493-495, 518-520)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )
        elif model_type in ["qwen2", "qwen3"]:
            # For qwen2/qwen3 models: no trust_remote_code (matches jiuge line 534-536)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)
        else:
            # Default: use trust_remote_code=True for other models
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_dir_path, trust_remote_code=True
            )

        eos_token_id = self.config_dict.get("eos_token_id")
        self.eos_token_id = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
        )

        # Create model with cpp backend
        print("Loading model with cpp backend...")
        self.model = AutoLlamaModel.from_pretrained(
            model_dir_path,
            device=self.device,
            dtype=self.dtype,
            backend=backend,
            distributed_config=DistConfig(ndev),
        )

        # Enable KV cache for generation
        self.model.use_cache = True

        # Load weights
        print("Loading model weights...")
        model_param_infini = get_model_state_dict(
            model_dir_path,
            device=self.device,
            dtype=self.dtype,
        )
        self.model.load_state_dict(model_param_infini)
        print("Model loaded successfully")

    def max_context_len(self):
        return self.config_dict.get("max_position_embeddings", 2048)

    def render_input_content(self, *args, **kwargs):
        """Render input content based on benchmark type"""
        if self.benchmark == "ceval":
            return self._render_ceval(*args, **kwargs)
        elif self.benchmark == "mmlu":
            return self._render_mmlu(*args, **kwargs)
        else:
            raise ValueError(f"Unknown benchmark: {self.benchmark}")

    def _render_ceval(self, conversation):
        """Render C-Eval conversation to input content"""
        return (
            self.tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            + "正确答案是"
        )

    def _render_mmlu(self, question, choices):
        """Render MMLU question and choices to input content"""
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        instruction = (
            "You are a multiple-choice question solver. "
            "Select the correct option and respond with only the letter A, B, C, or D."
        )
        prompt = f"{instruction}\n\nQuestion: {question}\n{choices_text}\nAnswer:"

        # Use chat template if available, otherwise return plain text
        if hasattr(self.tokenizer, 'apply_chat_template'):
            conversation = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": f"{question}\n{choices_text}\nAnswer:"}
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    conversation=conversation,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                return prompt
        return prompt

    def generate(self, *args, max_steps=500, topp_=1.0, topk_=1, temperature_=1.0):
        """Generate response based on benchmark type"""
        # Render input content
        input_content = self.render_input_content(*args)
        print(input_content, end="", flush=True)

        # Encode input
        tokens = self.encode_text(input_content)

        # Delegate to backend-specific generation implementation
        output_content, avg_time = self._generate_step(
            tokens, max_steps, topp_, topk_, temperature_
        )

        return output_content, avg_time

    def _generate_step(self, tokens, max_steps, topp_, topk_, temperature_):
        """
        InfiniLM cpp backend-specific generation implementation

        NOTE: Validation confirmed input configs are identical between backends.
        The issue was that manual generation loop called InferEngine.generate() which
        doesn't maintain KV cache. Solution: Use model's built-in generate() method
        which properly handles KV cache through GenerationMixin.
        """
        # Convert tokens to infinicore format
        input_ids_list = [tokens]
        input_ids = infinicore.from_list(input_ids_list, dtype=infinicore.int64).to(self.device)

        # Use model's built-in generate() method which properly handles KV cache
        # Pass sampling parameters (temperature, topk, topp) via kwargs
        output_tokens_list, output_content = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_steps,
            tokenizer=self.tokenizer,
            stop_on_eos=True,
            temperature=temperature_,
            topk=topk_,
            topp=topp_,
        )

        # Calculate average time (GenerationMixin doesn't return timing info)
        # We'll use a placeholder since the timing info isn't available
        print("\n")
        avg_time = 0.0  # GenerationMixin doesn't expose per-step timing
        print(f"Time per step: N/A (using GenerationMixin.generate)")

        return output_content, avg_time

    def destroy_model_instance(self):
        # Cleanup if needed
        del self.model
        print("Model destroyed")


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
        return ord(match.group(1)) - ord('A')
    else:
        match_num = re.search(r"\b([0-3])\b", output_upper)
        if match_num:
            return int(match_num.group(1))
    return None


def test():
    # Parse arguments manually to handle device flags properly
    if len(sys.argv) < 4:
        print(
            "Usage: python test_benchmark.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> --bench [ceval|mmlu] [--backend cpp] [--ndev N] [--subject SUBJECT] [--num_samples N] [--max_new_tokens N]"
        )
        sys.exit(1)

    # Parse device flag (first argument)
    device_flag = sys.argv[1]
    model_path = sys.argv[2]

    # Parse optional arguments
    backend = "cpp"
    ndev = 1
    benchmark = None
    subject = None  # For MMLU
    dataset_name = "middle_school_mathematics"  # For C-Eval
    num_samples = None
    max_new_tokens = 500

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--bench" and i + 1 < len(sys.argv):
            benchmark = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--ndev" and i + 1 < len(sys.argv):
            ndev = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--subject" and i + 1 < len(sys.argv):
            subject = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--dataset" and i + 1 < len(sys.argv):
            dataset_name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--num_samples" and i + 1 < len(sys.argv):
            num_samples = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--max_new_tokens" and i + 1 < len(sys.argv):
            max_new_tokens = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    if benchmark is None:
        print("Error: --bench argument is required. Choose 'ceval' or 'mmlu'")
        sys.exit(1)

    if benchmark not in ["ceval", "mmlu"]:
        print(f"Error: Unknown benchmark '{benchmark}'. Choose 'ceval' or 'mmlu'")
        sys.exit(1)

    # Parse device type
    device_type_str = "cpu"
    if device_flag == "--cpu":
        device_type_str = "cpu"
    elif device_flag == "--nvidia":
        device_type_str = "nvidia"
    elif device_flag == "--cambricon":
        device_type_str = "cambricon"
    elif device_flag == "--ascend":
        device_type_str = "ascend"
    elif device_flag == "--metax":
        device_type_str = "metax"
    elif device_flag == "--moore":
        device_type_str = "moore"
    elif device_flag == "--iluvatar":
        device_type_str = "iluvatar"
    elif device_flag == "--kunlun":
        device_type_str = "kunlun"
    elif device_flag == "--hygon":
        device_type_str = "hygon"
    else:
        print(
            "Usage: python test_benchmark.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> --bench [ceval|mmlu] [--backend cpp] [--ndev N] [--subject SUBJECT] [--num_samples N] [--max_new_tokens N]"
        )
        sys.exit(1)

    # Load dataset based on benchmark
    if benchmark == "ceval":
        # Load C-Eval dataset
        # https://huggingface.co/datasets/ceval/ceval-exam/tree/main/middle_school_geography
        print(f"Loading C-Eval dataset (dataset: {dataset_name})...")
        try:
            dataset = load_dataset(r"ceval/ceval-exam", name=dataset_name)
            samples = dataset["val"]
            # Convert Dataset to list if needed
            if hasattr(samples, 'to_list'):
                samples = samples.to_list()
            else:
                samples = list(samples)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Available datasets: middle_school_mathematics, high_school_history, high_school_chinese, high_school_physics, middle_school_geography, middle_school_physics")
            sys.exit(1)

    elif benchmark == "mmlu":
        # Load MMLU dataset
        # https://huggingface.co/datasets/cais/mmlu
        if subject is None:
            subject = "all"
        print(f"Loading MMLU dataset (subject: {subject})...")
        try:
            if subject == "all":
                dataset = load_dataset("cais/mmlu", "all")
                # Combine all subjects into a single dataset
                samples = []
                for subject_name in dataset.keys():
                    if subject_name in ["train", "validation", "test"]:
                        continue
                    # Convert Dataset to list
                    test_data = dataset[subject_name]["test"]
                    if hasattr(test_data, 'to_list'):
                        samples.extend(test_data.to_list())
                    else:
                        samples.extend(list(test_data))
            else:
                dataset = load_dataset("cais/mmlu", subject)
                test_data = dataset["test"]
                # Convert Dataset to list
                if hasattr(test_data, 'to_list'):
                    samples = test_data.to_list()
                else:
                    samples = list(test_data)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Available subjects: abstract_algebra, anatomy, astronomy, business_ethics, etc.")
            print("Use --subject all to load all subjects")
            sys.exit(1)

    print(f"Loaded {len(samples)} samples")

    # Limit number of samples if specified
    if num_samples is not None and num_samples > 0:
        original_count = len(samples)
        samples = samples[:num_samples]
        print(f"Limited to {len(samples)} samples for validation (from {original_count} total)")

    # Create model based on backend
    if backend != "010":
        model = InfiniLMBenchmark(model_path, device_type_str, ndev, backend, benchmark)
    else:
        print(f"test 010 backend by scripts/test_ceval.py")
        exit(0)

    # Test with first sample if available
    if len(samples) > 0:
        sample = samples[0]
        if benchmark == "ceval":
            input_content = f"'question':{sample['question']},'A': {sample['A']}, 'B':{sample['B']}, 'C': {sample['C']},'D': {sample['D']}。"
            test_conversation = [
                {
                    "role": "system",
                    "content": "请从question的A，B，C，D四个选项中选择正确的选项。例如，标准答案：A。",
                },
                {"role": "user", "content": input_content},
            ]
            test_output, _ = model.generate(test_conversation, max_steps=max_new_tokens, topp_=1.0, topk_=1, temperature_=1.0)
        elif benchmark == "mmlu":
            question = sample['question']
            choices = sample['choices']
            test_output, _ = model.generate(question, choices, max_steps=max_new_tokens, topp_=1.0, topk_=1, temperature_=1.0)
        print(f"\nTest output: {test_output}")

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
            output_content, avg_time = model.generate(
                conversation, max_steps=max_new_tokens, topp_=1.0, topk_=1, temperature_=1.0
            )
            is_correct = extract_answer_ceval(output_content, answer)
            answers_list.append({
                "id": sample.get("id", idx),
                "output_content": output_content,
                "answer": answer,
                "is_correct": is_correct
            })
            if benchmark == "ceval":
                print("标准答案：", answer)

        elif benchmark == "mmlu":
            question = sample['question']
            choices = sample['choices']
            answer_idx = sample['answer']  # MMLU answer is 0-3 index

            output_content, avg_time = model.generate(
                question, choices, max_steps=max_new_tokens, topp_=1.0, topk_=1, temperature_=1.0
            )

            predicted_answer = extract_answer_mmlu(output_content)

            # Convert answer index to letter for display
            answer_letter = chr(65 + answer_idx) if answer_idx < 4 else "?"
            predicted_letter = chr(65 + predicted_answer) if predicted_answer is not None and predicted_answer < 4 else "?"

            print(f"Sample {idx}: Correct answer: {answer_letter} ({answer_idx}), Predicted: {predicted_letter} ({predicted_answer})")

            answers_list.append({
                "id": idx,
                "output_content": output_content,
                "answer": answer_idx,
                "predicted": predicted_answer
            })

    model.destroy_model_instance()

    print("-------------------------------------------------------------")

    # Evaluate results
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
                predicted_letter = chr(65 + predicted) if predicted is not None and predicted < 4 else "?"
                print(f"id {id}: Wrong (correct: {answer_letter}, predicted: {predicted_letter})")

    accuracy = true_num / all_num if all_num > 0 else 0.0
    if benchmark == "ceval":
        print(f"成绩: {true_num}/{all_num}", accuracy)
    else:
        print(f"Accuracy: {true_num}/{all_num} = {accuracy:.2%}")


if __name__ == "__main__":
    test()
