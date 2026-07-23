import argparse
import csv
import json
import os
import re

from backends import InfiniLMBenchmark, TransformersBenchmark, VLLMBenchmark
from datasets import Dataset, load_dataset
from infinilm.base_config import BaseConfig


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
                    x in lower for x in ["-val", "validation", "dev"]
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
                "No MMLU cached data found for any subject. Please ensure datasets are cached."
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
                        "No MMLU data found for any subject in the list"
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
    cfg = BaseConfig()

    device_type_str = cfg.device

    # Normalize cache_dir and force offline when provided
    if cfg.cache_dir:
        cfg.cache_dir = os.path.expanduser(cfg.cache_dir)
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Step 1: Load dataset samples first
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60 + "\n")

    subject_samples = load_dataset_samples(cfg)

    if not subject_samples:
        print("No samples loaded. Exiting.")
        return

    # Step 2: Create model based on backend
    print("\n" + "=" * 60)
    print("STEP 2: LOADING MODEL")
    print("=" * 60 + "\n")

    device_str = cfg.get_device_str(device_type_str)
    if cfg.backend in {"transformers", "torch"}:
        model = TransformersBenchmark(cfg.model, device_str, cfg.tp, cfg.bench)
    elif cfg.backend == "vllm":
        model = VLLMBenchmark(cfg.model, device_str, cfg.tp, cfg.bench)
    elif cfg.backend in {"infinilm", "cpp", "python"}:
        model = InfiniLMBenchmark(
            model_dir_path=cfg.model,
            device_type_str=device_str,
            tensor_parallel_size=cfg.tp,
            benchmark=cfg.bench,
            enable_paged_attn=cfg.enable_paged_attn,
            enable_graph=cfg.enable_graph,
            attn_backend=cfg.attn,
            weight_load_mode=cfg.weight_load_mode,
        )
    else:
        raise ValueError(f"Unsupported backend: {cfg.backend}")
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
            model, samples, cfg.bench, cfg.max_new_tokens, subject_name
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
    if cfg.bench == "ceval":
        print(
            f"Overall 成绩: {overall_correct}/{overall_total} = {overall_accuracy:.2%}"
        )
    else:
        print(
            f"Overall Accuracy: {overall_correct}/{overall_total} = {overall_accuracy:.2%}"
        )

    print(f"Total Latency: {model.total_time:.2f} seconds")
    print(f"Total Tokens Processed: {model.total_tokens} tokens")
    if model.total_time > 0:
        print(
            f"Overall Throughput: {model.total_tokens / model.total_time:.2f} tokens/s"
        )

    # Write CSV if output path is specified
    if cfg.output_csv:
        print(f"\nWriting results to CSV: {cfg.output_csv}")
        with open(cfg.output_csv, "w", newline="", encoding="utf-8") as csvfile:
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
        print(f"CSV file written successfully: {cfg.output_csv}")


if __name__ == "__main__":
    main()
