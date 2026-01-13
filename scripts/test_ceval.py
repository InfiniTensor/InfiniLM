import sys
from jiuge import *
from datasets import load_dataset


class JiugeForCeval(JiugeForCauslLM):
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        super().__init__(model_dir_path, device, ndev, max_tokens)
        pass

    def generate(self, conversation, max_steps, topp_=1.0, topk_=0, temperature_=1.0, repetition_penalty_=1.03):
        # Align with launch_server.py: use apply_chat_template with enable_thinking=False
        template_params = {
            "conversation": conversation,
            "add_generation_prompt": True,
            "tokenize": False,
            "enable_thinking": False  # Disable thinking mode
        }
        input_content = (
            self.tokenizer.apply_chat_template(**template_params)
            + "正确答案是"
        )

        print(input_content, end="", flush=True)

        tokens = self.tokenizer.encode(input_content)
        infer_task = InferTask(
            0,
            tokens,
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            self.eos_token_id,
            repetition_penalty_,
        )
        infer_task.bind_kvcache(KVCache(self))

        steps = 0
        total_time = 0
        # Collect all tokens first, then decode at once to preserve UTF-8 sequences (aligned with launch_server)
        output_tokens = []

        for step_i in range(max_steps):
            start_time = time.time()
            step_output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1

            token = step_output_tokens[0]

            # Check for EOS before adding to buffer
            if token is None or token in self.eos_token_id:
                break

            output_tokens.append(token)
            infer_task.next(token)

            if step_i > 0:
                total_time += end_time - start_time

        # Decode all tokens at once to preserve multi-byte UTF-8 sequences (aligned with launch_server non-streaming)
        if output_tokens:
            output_content = self.tokenizer.decode(output_tokens, skip_special_tokens=False).strip()
            print(output_content, end="", flush=True)
        else:
            output_content = ""

        print("\n")
        avg_time = total_time * 1000 / (steps - 1 + 1e-9) if steps > 1 else 0
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python test_ceval.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU
    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif sys.argv[1] == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif sys.argv[1] == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif sys.argv[1] == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif sys.argv[1] == "--moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
    elif sys.argv[1] == "--iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    elif sys.argv[1] == "--kunlun":
        device_type = DeviceType.DEVICE_TYPE_KUNLUN
    elif sys.argv[1] == "--hygon":
        device_type = DeviceType.DEVICE_TYPE_HYGON
    else:
        print(
            "Usage: python test_ceval.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    # Full list of subjects from https://huggingface.co/datasets/ceval/ceval-exam/tree/main
    ALL_SUBJECTS = [
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
        "veterinary_medicine"
    ]

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = JiugeForCeval(model_path, device_type, ndev)

    # Overall statistics across all subjects
    overall_true_num = 0
    overall_all_num = 0
    subject_results = {}

    # Test each subject
    for subject in ALL_SUBJECTS:
        print("=" * 80)
        print(f"Testing subject: {subject}")
        print("=" * 80)

        try:
            # Load dataset for this subject
            dataset = load_dataset(r"ceval/ceval-exam", name=subject)
            samples = dataset["test"]

            answers_list = []
            for sample in samples:
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
                    conversation, 1000, topp_=0.1, topk_=0, temperature_=0.9
                )
                print("标准答案：", answer)
                answers_list.append(
                    {"id": sample["id"], "output_content": output_content, "answer": answer}
                )

            # Calculate accuracy for this subject
            true_num = 0
            all_num = 0
            for cont in answers_list:
                id = cont["id"]
                output = cont["output_content"]
                answer = cont["answer"]

                all_num = all_num + 1
                position = 0
                ABCD = output[position : position + 2]
                if answer in ABCD:
                    true_num = true_num + 1
                    print(f"id {id} : ", "正确")
                else:
                    print(f"id {id}: ", "错误")

            accuracy = true_num / all_num if all_num > 0 else 0.0
            print(f"\nSubject: {subject}")
            print(f"成绩: {true_num}/{all_num} = {accuracy:.4f} ({accuracy*100:.2f}%)")

            # Store results
            subject_results[subject] = {
                "true_num": true_num,
                "all_num": all_num,
                "accuracy": accuracy
            }
            overall_true_num += true_num
            overall_all_num += all_num

        except Exception as e:
            print(f"Error testing subject {subject}: {e}")
            subject_results[subject] = {
                "true_num": 0,
                "all_num": 0,
                "accuracy": 0.0,
                "error": str(e)
            }

    # Destroy model instance
    model.destroy_model_instance()

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - All Subjects")
    print("=" * 80)

    # Print results for each subject
    for subject, result in subject_results.items():
        if "error" in result:
            print(f"{subject:45s}: ERROR - {result['error']}")
        else:
            print(f"{subject:45s}: {result['true_num']:4d}/{result['all_num']:4d} = {result['accuracy']*100:6.2f}%")

    # Print overall statistics
    overall_accuracy = overall_true_num / overall_all_num if overall_all_num > 0 else 0.0
    print("=" * 80)
    print(f"Overall: {overall_true_num}/{overall_all_num} = {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    test()
