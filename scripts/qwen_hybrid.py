from typing import List, Sequence
import math
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import torch
import transformers

from libinfinicore_infer import (
    QwenHybridModel,
    QwenHybridMetaCStruct,
    DataType,
    DeviceType,
    KVCacheCStruct,
    MambaCacheCStruct,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref

torch.set_default_device("cpu")


class QwenHybridMetaFromConfig(QwenHybridMetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        if config["torch_dtype"] == "float16":
            dt_ = DataType.INFINI_DTYPE_F16
        elif config["torch_dtype"] == "float32":
            dt_ = DataType.INFINI_DTYPE_F32
        elif config["torch_dtype"] == "bfloat16":
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        eos_token_id = (
            config["eos_token_id"][-1]
            if type(config["eos_token_id"]) == list
            else config["eos_token_id"]
        )

        super().__init__(
            # common
            dtype=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            end_token=eos_token_id,
            # mha
            nh=config["num_attention_heads"],
            nkvh=(
                config["num_key_value_heads"]
                if "num_key_value_heads" in config
                else config["num_attention_heads"]
            ),
            dh=config.get(
                "head_dim", config["hidden_size"] // config["num_attention_heads"]
            ),
            theta=(config["rope_theta"] if "rope_theta" in config else 100000.0),
            use_qk_norm=config.get("use_qk_norm", True),
            # linear attn
            l_conv_kernel_dim=config.get("linear_conv_kernel_dim", 0),
            l_expand=config.get("linear_expand_v", 0),
            l_n_k_head=config.get("linear_num_key_heads", 0),
            l_k_dim=config.get("linear_key_head_dim", 0),
            l_n_v_head=config.get("linear_num_value_heads", 0),
            l_v_dim=config.get("linear_value_head_dim", 0),
            # moe
            nexperts=config.get("num_experts", 0),
            kexperts=config.get("num_experts_per_tok", 0),
            shared_di=config.get("shared_expert_intermediate_size", 5632),
            moe_di=config.get("moe_intermediate_size", 0),
            norm_topk_prob=config.get("norm_topk_prob", False),
        )
        self.torch_dtype_logits = dtype


class QwenHybridBatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.mamba_cache_ptrs = [t.mamba_cache() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.mamba_caches = (POINTER(MambaCacheCStruct) * self.nreq)(
            *self.mamba_cache_ptrs
        )

        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.mamba_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )


class QwenHybridForCausalLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):

        load_start_time = time.time()
        print(f"Creating model on {ndev} devices...")
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        self.dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.ndev = ndev
        self.device = device
        self.meta = QwenHybridMetaFromConfig(config, max_tokens=max_tokens)

        self.model = QwenHybridModel()

        self.weights = self.model.create_weights(
            byref(self.meta),
            self.device,
            ndev,
            self.dev_ids,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

        load_start_time = time.time()
        print("Loading model weights to host...")

        self.load_all_safetensors_from_dir(os.path.join(model_dir_path))

        self.model_instance = self.model.create_model(
            byref(self.meta),
            self.weights,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def load_all_safetensors_from_dir(self, dir_path_: str):
        dir_path_ = Path(dir_path_)
        for file in sorted(dir_path_.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "mtp" in key:
                        continue
                    tensor = f.get_tensor(key)
                    if (
                        "q_norm" in key
                        or "k_norm" in key
                        or "input_layernorm" in key
                        or "post_attention_layernorm" in key
                        or "model.norm" in key
                    ):
                        # print(f"Add 1 to qk norm weight: {key}")
                        tensor = tensor + 1
                    elif "A_log" in key:
                        # print(f"Transforming A_log weight: {key}")
                        tensor = -tensor.float().exp().to(tensor.dtype)
                    elif (
                        "k_proj" in key or "v_proj" in key
                    ) and self.ndev > self.meta.nkvh:
                        repeat_times = self.ndev // self.meta.nkvh
                        tensor = torch.cat([tensor for _ in range(repeat_times)], dim=0)
                        tensor = (
                            tensor.view(repeat_times, self.meta.nkvh, -1)
                            .permute(1, 0, 2)
                            .contiguous()
                        )

                    elif "conv1d" in key:
                        n_k_h = self.meta.l_n_k_head
                        n_v_h = self.meta.l_n_v_head
                        k_d = self.meta.l_k_dim
                        v_d = self.meta.l_v_dim
                        ndev = self.ndev
                        block1, block2, block3 = torch.split(
                            tensor, [n_k_h * k_d, n_k_h * k_d, n_v_h * v_d], dim=0
                        )
                        blocks = []
                        for idev in range(ndev):
                            s_12 = block1.shape[0] // ndev
                            s_3 = block3.shape[0] // ndev
                            blocks.append(block1[idev * s_12 : (idev + 1) * s_12, :, :])
                            blocks.append(block2[idev * s_12 : (idev + 1) * s_12, :, :])
                            blocks.append(block3[idev * s_3 : (idev + 1) * s_3, :, :])
                        tensor = torch.cat(blocks, dim=0).contiguous()

                    self.model.load_weight(self.weights, key, tensor.data_ptr())

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return self.model.create_kv_cache(
            self.meta.nlayer // 4,  # Full attn every 4th layers
            self.meta.dctx,
            self.meta.nkvh,
            self.meta.dh,
            self.meta.dh,
            self.meta.dtype,
            self.device,
            self.dev_ids,
            self.ndev,
        )

    def drop_kv_cache(self, kv_cache):
        self.model.drop_kv_cache(kv_cache)

    def create_mamba_cache(self):
        return self.model.create_mamba_cache(
            (self.meta.nlayer // 4) * 3,
            self.meta.l_conv_kernel_dim,
            self.meta.l_k_dim,
            self.meta.l_v_dim,
            self.meta.l_n_k_head,
            self.meta.l_n_v_head,
            self.meta.dtype,
            self.device,
            self.dev_ids,
            self.ndev,
        )

    def drop_mamba_cache(self, mamba_cache):
        self.model.drop_mamba_cache(mamba_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = QwenHybridBatchedTask(tasks)
        self.model.infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        input_content = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,
            tokenize=False,
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
        infer_task.bind_mamba_cache(self.create_mamba_cache())

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
        self.drop_mamba_cache(infer_task._mamba_cache)
        return output_content, avg_time

    def perplexity(self, test_sequences: List[Sequence[int]], batch_size=10):
        tasks = [
            InferTask(i, [], self.max_context_len(), 1.0, 1, 1.0, self.eos_token_id)
            for i in range(batch_size)
        ]
        kv_caches = [KVCache(self) for _ in range(batch_size)]

        nll = 0.0
        total_len = 0

        for i in range(0, len(test_sequences), batch_size):
            batch_id = 0
            true_tokens = []
            while batch_id < batch_size and batch_id + i < len(test_sequences):
                input_tokens = test_sequences[i + batch_id][:-1]
                true_tokens.extend(test_sequences[i + batch_id][1:])
                tasks[batch_id].tokens = input_tokens
                tasks[batch_id].bind_kvcache(kv_caches[batch_id])
                batch_id += 1

            batch_inputs = QwenHybridBatchedTask(tasks[:batch_id])
            logits = torch.zeros(
                (batch_inputs.ntok, self.meta.dvoc), dtype=self.meta.torch_dtype_logits
            )
            self.model.forward_batch(
                self.model_instance,
                batch_inputs.tokens,
                batch_inputs.ntok,
                batch_inputs.req_lens,
                batch_inputs.nreq,
                batch_inputs.req_pos,
                batch_inputs.kv_caches,
                logits.data_ptr(),
            )

            logits = logits.float()
            token_ids = torch.tensor(true_tokens, dtype=torch.int64)  # [ntok,]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (ntok, vocab)
            token_logprobs = log_probs[
                torch.arange(batch_inputs.ntok), token_ids
            ]  # (ntok,)

            start = 0
            for l in batch_inputs.req_lens_list:
                nll += -token_logprobs[start : start + l].sum().item()
                start += l
            total_len += token_logprobs.numel()

        for task in tasks:
            task.release_kvcache()

        return math.exp(nll / total_len)

    def destroy_model_instance(self):
        self.model.destroy_model(self.model_instance)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python qwen_hybrid.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
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
    else:
        print(
            "Usage: python qwen_hybrid.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = QwenHybridForCausalLM(model_path, device_type, ndev)
    model.generate("山东最高的山是？", 50)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
