import sys
import os
import argparse
import time
from jiuge import *
from datasets import load_dataset
import infinicore
import infinilm
from infinilm.models.llama import AutoLlamaModel
from infinilm.modeling_utils import get_model_state_dict
from infinilm.distributed import DistConfig
from abc import ABC, abstractmethod


class BaseForCeval(ABC):
    """Base class for C-Eval benchmark with common tokenizer and generation utilities"""

    def render_input_content(self, conversation):
        """Render conversation to input content - reused across backends"""
        return (
            self.tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            + "正确答案是"
        )

    def encode_text(self, text):
        """Encode text to token IDs - reused across backends"""
        return self.tokenizer.encode(text)

    def decode_token(self, token_id):
        """Decode token ID to text - reused across backends"""
        return self.tokenizer.decode(token_id)

    def generate(self, conversation, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        """
        Common generation loop - delegates to backend-specific _generate_step
        """
        # Reuse render_input_content method
        input_content = self.render_input_content(conversation)

        print(input_content, end="", flush=True)

        # Encode input - reuse encode_text method
        tokens = self.encode_text(input_content)


        # Delegate to backend-specific generation implementation
        output_content, avg_time = self._generate_step(
            tokens, max_steps, topp_, topk_, temperature_
        )

        return output_content, avg_time

    @abstractmethod
    def _generate_step(self, tokens, max_steps, topp_, topk_, temperature_):
        """
        Backend-specific generation implementation.
        Should return (output_content, avg_time)
        """
        pass


class JiugeForCeval(JiugeForCauslLM, BaseForCeval):
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        super().__init__(model_dir_path, device, ndev, max_tokens)
        pass

    def generate(self, conversation, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        """
        Override to use BaseForCeval.generate instead of JiugeForCauslLM.generate.
        Explicitly call BaseForCeval's generate to bypass JiugeForCauslLM's generate method.
        """
        # Directly call BaseForCeval.generate to bypass JiugeForCauslLM.generate
        # This ensures we use the conversation-based API instead of input_content-based API
        return BaseForCeval.generate(self, conversation, max_steps, topp_, topk_, temperature_)

    def _generate_step(self, tokens, max_steps, topp_, topk_, temperature_):
        """Jiuge backend-specific generation implementation"""
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
            output_str = self.decode_token(output_tokens[0])
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


class InfiniLMForCeval(BaseForCeval):
    """Wrapper class for InfiniLM cpp backend to match JiugeForCauslLM interface"""

    def __init__(self, model_dir_path, device_type_str="cpu", ndev=1, backend="cpp", max_tokens=None):
        import transformers

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


def test():
    # Parse arguments manually to handle device flags properly
    if len(sys.argv) < 3:
        print(
            "Usage: python test_ceval.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> [--backend cpp] [--ndev N]"
        )
        sys.exit(1)

    # Parse device flag (first argument)
    device_flag = sys.argv[1]
    model_path = sys.argv[2]

    # Parse optional arguments
    backend = "jiuge"
    ndev = 1

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--backend" and i + 1 < len(sys.argv):
            backend = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--ndev" and i + 1 < len(sys.argv):
            ndev = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    # Parse device type
    device_type = DeviceType.DEVICE_TYPE_CPU
    device_type_str = "cpu"
    if device_flag == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
        device_type_str = "cpu"
    elif device_flag == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
        device_type_str = "nvidia"
    elif device_flag == "--cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
        device_type_str = "cambricon"
    elif device_flag == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
        device_type_str = "ascend"
    elif device_flag == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
        device_type_str = "metax"
    elif device_flag == "--moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
        device_type_str = "moore"
    elif device_flag == "--iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
        device_type_str = "iluvatar"
    elif device_flag == "--kunlun":
        device_type = DeviceType.DEVICE_TYPE_KUNLUN
        device_type_str = "kunlun"
    elif device_flag == "--hygon":
        device_type = DeviceType.DEVICE_TYPE_HYGON
        device_type_str = "hygon"
    else:
        print(
            "Usage: python test_ceval.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore | --iluvatar | --kunlun | --hygon] <path/to/model_dir> [--backend cpp] [--ndev N]"
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

    # Create model based on backend
    if backend != "010":
        model = InfiniLMForCeval(model_path, device_type_str, ndev, backend)
    else:
        model = JiugeForCeval(model_path, device_type, ndev)

    if len(samples) > 0:
        sample = samples[0]
        input_content = f"'question':{sample['question']},'A': {sample['A']}, 'B':{sample['B']}, 'C': {sample['C']},'D': {sample['D']}。"
        test_conversation = [
            {
                "role": "system",
                "content": "请从question的A，B，C，D四个选项中选择正确的选项。例如，标准答案：A。",
            },
            {"role": "user", "content": input_content},
        ]

        # Validate input rendering and tokenization for current backend
        rendered = model.render_input_content(test_conversation)
        tokens = model.encode_text(rendered)


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
