from typing import List, Sequence

from sympy import true
from libinfinicore_infer.qwen3_moe import (
    MoEMetaCStruct,
    WeightsCStruct,
    DataType,
    DeviceType,
    KVCacheCStruct,
    create_model,
    destroy_model,
    create_kv_cache,
    drop_kv_cache,
    infer_batch,
    forward_batch,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import math
import torch
import transformers

torch.set_default_device("cpu")


class Qwen3MoEMeta(MoEMetaCStruct):
    def __init__(self, config: dict, dtype=torch.float16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config["num_key_value_heads"] if "num_key_value_heads" in config else config["num_attention_heads"],
            dh=config["head_dim"] if "head_dim" in config else (config["hidden_size"] // config["num_attention_heads"]),
            di=config["intermediate_size"],
            dctx=config["max_position_embeddings"] if max_tokens is None else max_tokens,
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"] if "rope_theta" in config else 100000.0,
            end_token=config["eos_token_id"],
            #
            _moe_intermediate_size=config["moe_intermediate_size"],
            _shared_expert_intermediate_size=config["shared_expert_intermediate_size"] if "shared_expert_intermediate_size" in config else 0,
            _num_experts=config["num_experts"],
            _num_experts_per_tok=config["num_experts_per_tok"],
            _norm_topk_prob=config["norm_topk_prob"],
        )
        self.torch_dtype_logits = dtype


class Qwen3MoEWeights(WeightsCStruct):
    def __init__(self,
                 meta: Qwen3MoEMeta,
                 state_dict: dict,
                 torch_dt_mat=torch.float16,
                 torch_dt_norm=torch.float32,
                 ndev=1,
                 transpose_weight: bool = True,
                 ):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        num_experts = meta._num_experts

        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0

        torch_dt_logits = meta.torch_dtype_logits

        _moe_intermediate_size = meta._moe_intermediate_size
        _shared_expert_intermediate_size = meta._shared_expert_intermediate_size

        _num_experts_per_tok = meta._num_experts_per_tok
        _norm_topk_prob = meta._norm_topk_prob

        super().__init__(nlayer, num_experts, nh, nkvh, d, di, dh, ndev,
                         torch_dt_mat, torch_dt_logits, torch_dt_norm,
                         transpose_weight,
                         _moe_intermediate_size, _shared_expert_intermediate_size, _num_experts_per_tok, _norm_topk_prob,
                         state_dict)


class BatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
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
            self.temperaturas,
            self.topks,
            self.topps,
        )


# ---------------------------
# ---------------------------
# ---------------------------
def load_all_safetensors_from_dir(dir_path_: str):
    tensors_ = {}
    dir_path_ = Path(dir_path_)
    for file in sorted(dir_path_.glob("*.safetensors")):
        data_ = safetensors.safe_open(file, "pt")
        for name_ in data_.keys():
            tensors_[name_] = data_.get_tensor(name_)
    return tensors_


def load_config_json(dir_path_: str):
    with open(os.path.join(dir_path_, "config.json"), "r") as f:
        config = json.load(f)
    return config


class Qwen3MoEForCauslLM:
    def __init__(
            self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        print("Loading model weights to host...")
        load_start_time = time.time()

        self.config = load_config_json(model_dir_path)
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = [eos_token_id] if type(eos_token_id) == int else eos_token_id

        transpose_weight = (
                device != DeviceType.DEVICE_TYPE_ASCEND
        )  # y = xW is faster than y=xW^T on Ascend

        if "qwen3_moe" == self.config["model_type"]:
            state_dict = load_all_safetensors_from_dir(model_dir_path)

            self.meta = Qwen3MoEMeta(self.config, max_tokens=max_tokens)
            self.weights = Qwen3MoEWeights(
                self.meta,
                state_dict,
                ndev=ndev,
                transpose_weight=transpose_weight,
            )
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)
        else:
            raise ValueError("Unsupported model architecture")

        load_end_time = time.time()
        print(f"Qwen3MoEWeights, Time used: {load_end_time - load_start_time:.3f}s")

        print(f"Creating model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.model_instance = create_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )
        load_end_time = time.time()
        print(f"create_model Time used: {load_end_time - load_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return create_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        drop_kv_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = BatchedTask(tasks)
        infer_batch(
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
        # print("tokens: ", tokens)

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

            batch_inputs = BatchedTask(tasks[:batch_id])
            logits = torch.zeros(
                (batch_inputs.ntok, self.meta.dvoc), dtype=self.meta.torch_dtype_logits
            )
            forward_batch(
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
                nll += -token_logprobs[start: start + l].sum().item()
                start += l
            total_len += token_logprobs.numel()

        for task in tasks:
            task.release_kvcache()

        return math.exp(nll / total_len)

    def destroy_model_instance(self):
        destroy_model(self.model_instance)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print("Usage: python qwen3_moe.py  --nvidia <path/to/model_dir> [n_device]")
        sys.exit(1)
    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_NVIDIA
    if sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif sys.argv[1] == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    else:
        print("Usage: python qwen3_moe.py --nvidia <path/to/model_dir> [n_device]")
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = Qwen3MoEForCauslLM(model_path, device_type, ndev)
    model.generate("山东最高的山是？", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
