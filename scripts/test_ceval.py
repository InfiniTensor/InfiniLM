import sys
from jiuge import *
from datasets import load_dataset


class JiugeForCeval(JiugeForCauslLM):
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        super().__init__(model_dir_path, device, ndev, max_tokens)
        pass

    def generate(self, conversation, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        input_content = (
            self.tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
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
        )
        infer_task.bind_kvcache(KVCache(self))

        steps = 0
        total_time = 0
        output_content = ""

        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            end_time = time.time()
            steps += 1
            output_str = self.tokenizer.decode(output_tokens[0])
            output_content += output_str
            print(output_str, end="", flush=True)
            if output_tokens[0] in self.eos_token_id:
                break
            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1 + 1e-9)
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

    # https://huggingface.co/datasets/ceval/ceval-exam/tree/main/middle_school_geography

    dataset = load_dataset(r"ceval/ceval-exam", name="middle_school_mathematics")
    # dataset = load_dataset(r"ceval/ceval-exam", name="high_school_history")
    # dataset = load_dataset(r"ceval/ceval-exam", name="high_school_chinese")
    # dataset = load_dataset(r"ceval/ceval-exam", name="high_school_physics")
    # dataset = load_dataset(r"ceval/ceval-exam", name="middle_school_geography")
    # dataset = load_dataset(r"ceval/ceval-exam", name="middle_school_physics")

    samples = dataset["val"]
    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = JiugeForCeval(model_path, device_type, ndev)

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
            conversation, 500, topp_=1.0, topk_=1, temperature_=1.0
        )
        print("标准答案：", answer)
        answers_list.append(
            {"id": sample["id"], "output_content": output_content, "answer": answer}
        )

    model.destroy_model_instance()

    print("-------------------------------------------------------------")

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

    print(f"成绩: {true_num}/{all_num}", true_num / all_num)


if __name__ == "__main__":
    test()
