import sys
import os
import argparse
import time
import re
import csv
from datasets import load_dataset, Dataset
import infinicore
import infinilm
from infinilm.models.llama import AutoLlamaModel
from infinilm.modeling_utils import load_model_state_dict_by_file
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
        load_model_state_dict_by_file(
            self.model,
            model_dir_path,
            dtype=self.dtype,
        )
        # model_param_infini = get_model_state_dict(
        #     model_dir_path,
        #     device=self.device,
        #     dtype=self.dtype,
        # )
        # self.model.load_state_dict(model_param_infini)
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
        input_ids = infinicore.from_list(input_ids_list)

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
            output_content, avg_time = model.generate(
                conversation, max_steps=max_new_tokens, topp_=1.0, topk_=1, temperature_=1.0
            )
            is_correct = extract_answer_ceval(output_content, answer)
            answers_list.append({
                "id": sample.get("id", idx),
                "output_content": output_content,
                "answer": answer,
                "is_correct": is_correct,
                "subject": subject_name
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
                "predicted": predicted_answer,
                "subject": subject_name
            })

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
                predicted_letter = chr(65 + predicted) if predicted is not None and predicted < 4 else "?"
                print(f"id {id}: Wrong (correct: {answer_letter}, predicted: {predicted_letter})")

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
        "answers_list": answers_list
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
                if split == "val" and not any(x in lower for x in ["val", "validation", "dev"]):
                    continue
                if split == "all" and not any(x in lower for x in ["val", "validation", "dev", "test"]):
                    continue
                try:
                    ds = Dataset.from_file(os.path.join(root, fname))
                    records.extend(ds.to_list())
                except Exception:
                    continue
        if records:
            return records

    # If cache_dir provided and nothing loaded, fail without network
    raise FileNotFoundError(f"CEval cached data not found for subject '{subject_name}' with splits {split_names}")


def _load_mmlu_from_cache(cache_dir, subject_name, split):
    """
    Load MMLU data from local cache avoiding network calls.
    Scans cached Arrow files under cache_dir/cais___mmlu and filters by split.
    """
    def load_one(subj):
        split_names = (
            ["test"]
            if split == "test"
            else ["validation", "dev"]
            if split == "val"
            else ["validation", "dev", "test"]
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
                if split == "val" and not any(x in lower for x in ["validation", "dev"]):
                    continue
                if split == "all" and not any(x in lower for x in ["validation", "dev", "test"]):
                    continue
                try:
                    ds = Dataset.from_file(os.path.join(root, fname))
                    records.extend(ds.to_list())
                except Exception:
                    continue
        if records:
            return records
        raise FileNotFoundError(f"MMLU cached data not found for subject '{subj}' with splits {split_names}")

    if subject_name == "all":
        base_dir_hf = os.path.join(cache_dir, "cais___mmlu")
        subjects = []
        if os.path.isdir(base_dir_hf):
            subjects = [
                d for d in os.listdir(base_dir_hf) if os.path.isdir(os.path.join(base_dir_hf, d))
            ]
        if not subjects:
            return load_one("all"), "all"

        all_samples = []
        for subj in subjects:
            try:
                all_samples.extend(load_one(subj))
            except FileNotFoundError:
                continue
        if not all_samples:
            raise FileNotFoundError(
                f"No MMLU cached data found under '{base_dir_hf}'. Please ensure datasets are cached."
            )
        return all_samples, "all"

    return load_one(subject_name), subject_name


def test():
    # Parse arguments manually to handle device flags properly
    if len(sys.argv) < 4:
        print(
            "Usage: python test_benchmark.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> --bench [ceval|mmlu] [--backend cpp] [--ndev N] [--subject SUBJECT] [--split {test|val|all}] [--num_samples N] [--max_new_tokens N] [--output_csv PATH] [--cache_dir PATH]"
        )
        sys.exit(1)

    # Parse device flag (first argument)
    device_flag = sys.argv[1]
    model_path = sys.argv[2]

    # Parse optional arguments
    backend = "cpp"
    ndev = 1
    benchmark = None
    subject = "all"  # Shared for both C-Eval and MMLU, can be comma-separated
    split = "test"   # test | val | all
    num_samples = None
    max_new_tokens = 500
    output_csv = None
    cache_dir = None

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
        elif sys.argv[i] == "--split" and i + 1 < len(sys.argv):
            split = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--num_samples" and i + 1 < len(sys.argv):
            num_samples = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--max_new_tokens" and i + 1 < len(sys.argv):
            max_new_tokens = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--output_csv" and i + 1 < len(sys.argv):
            output_csv = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--cache_dir" and i + 1 < len(sys.argv):
            cache_dir = sys.argv[i + 1]
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
            "Usage: python test_benchmark.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> --bench [ceval|mmlu] [--backend cpp] [--ndev N] [--subject SUBJECT] [--num_samples N] [--max_new_tokens N] [--output_csv PATH] [--cache_dir PATH]"
        )
        sys.exit(1)

    # Normalize cache_dir and force offline when provided
    if cache_dir:
        cache_dir = os.path.expanduser(cache_dir)
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Parse comma-separated subjects
    if split not in ["test", "val", "all"]:
        print("Error: --split must be one of: test, val, all")
        sys.exit(1)

    if subject and subject != "all":
        subject_list = [s.strip() for s in subject.split(",")]
    else:
        subject_list = ["all"]

    # Create model based on backend (create once, reuse for all subjects)
    if backend != "010":
        model = InfiniLMBenchmark(model_path, device_type_str, ndev, backend, benchmark)
    else:
        print(f"test 010 backend by scripts/test_ceval.py")
        exit(0)

    # Define helper functions for loading datasets
    if benchmark == "ceval":
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
            if cache_dir:
                return _load_ceval_from_cache(cache_dir, subj, split, ceval_subjects)
            # online fallback via HF load_dataset
            if split == "all":
                records = []
                for split_name in ["val", "test"]:
                    try:
                        ds = load_dataset(r"ceval/ceval-exam", name=subj, split=split_name)
                        records.extend(ds.to_list())
                    except Exception:
                        continue
                if records:
                    return records
                raise FileNotFoundError(f"No ceval splits found online for subject {subj}")
            hf_split = "test" if split == "test" else "val"
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
                    raise ValueError(f"Unknown C-Eval subject '{subj_name}'. Available subjects: {', '.join(ceval_subjects)}")
                return _load_ceval_subject(subj_name), subj_name

    elif benchmark == "mmlu":
        def _load_mmlu_subject(subj):
            print(f"Loading MMLU dataset (subject: {subj})...")
            if cache_dir:
                return _load_mmlu_from_cache(cache_dir, subj, split)
            if subj == "all":
                dataset = load_dataset("cais/mmlu", "all")
                samples = []
                splits_to_load = ["test"] if split == "test" else ["validation"] if split == "val" else ["validation", "test"]
                for subject_name in dataset.keys():
                    if subject_name in ["train", "validation", "test"]:
                        continue
                    for sp in splits_to_load:
                        if sp not in dataset[subject_name]:
                            continue
                        split_data = dataset[subject_name][sp]
                        if hasattr(split_data, 'to_list'):
                            samples.extend(split_data.to_list())
                        else:
                            samples.extend(list(split_data))
                return samples, "all"
            else:
                splits_to_load = ["test"] if split == "test" else ["validation"] if split == "val" else ["validation", "test"]
                records = []
                for sp in splits_to_load:
                    try:
                        dataset = load_dataset("cais/mmlu", subj, split=sp)
                        if hasattr(dataset, 'to_list'):
                            records.extend(dataset.to_list())
                        else:
                            records.extend(list(dataset))
                    except Exception:
                        continue
                if not records:
                    raise FileNotFoundError(f"MMLU subject {subj} split(s) {splits_to_load} not found")
                return records, subj

        def load_subject_samples(subj_name):
            return _load_mmlu_subject(subj_name)

    # Evaluate each subject separately
    all_results = []

    for subj in subject_list:
        print(f"\n{'='*60}")
        print(f"Evaluating subject: {subj}")
        print(f"{'='*60}\n")

        try:
            samples, actual_subj_name = load_subject_samples(subj)
            print(f"Loaded {len(samples)} samples for subject: {actual_subj_name}")

            # Limit number of samples if specified
            if num_samples is not None and num_samples > 0:
                original_count = len(samples)
                samples = samples[:num_samples]
                print(f"Limited to {len(samples)} samples for validation (from {original_count} total)")

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
                print(f"\nTest output: {test_output}\n")

            # Evaluate samples for this subject
            result = evaluate_samples(model, samples, benchmark, max_new_tokens, actual_subj_name)
            all_results.append(result)
            print(f"\nSubject '{actual_subj_name}' completed: {result['correct']}/{result['total']} = {result['accuracy']:.2%}")

        except Exception as e:
            print(f"Error evaluating subject '{subj}': {e}")
            continue

    model.destroy_model_instance()

    # Calculate overall results
    overall_correct = sum(r['correct'] for r in all_results)
    overall_total = sum(r['total'] for r in all_results)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

    print(f"\n{'='*60}")
    print("OVERALL RESULTS")
    print(f"{'='*60}")
    if benchmark == "ceval":
        print(f"Overall 成绩: {overall_correct}/{overall_total} = {overall_accuracy:.2%}")
    else:
        print(f"Overall Accuracy: {overall_correct}/{overall_total} = {overall_accuracy:.2%}")

    # Write CSV if output path is specified
    if output_csv:
        print(f"\nWriting results to CSV: {output_csv}")
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Subject', 'Correct', 'Total', 'Accuracy'])
            for result in all_results:
                writer.writerow([result['subject'], result['correct'], result['total'], f"{result['accuracy']:.4f}"])
            writer.writerow(['Overall', overall_correct, overall_total, f"{overall_accuracy:.4f}"])
        print(f"CSV file written successfully: {output_csv}")


if __name__ == "__main__":
    test()
