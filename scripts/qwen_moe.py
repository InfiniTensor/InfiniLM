from typing import List, Sequence
# 1. Import the new initialization function and necessary classes
from libinfinicore_infer import (
    QwenMoeMetaCStruct,
    QwenMoeWeightsCStruct,
    KVCacheCStruct,
    DataType,
    DeviceType,
    initialize_moe_apis
)
# 2. Call the function to get the MoE APIs (real or mock)
create_qwen_moe_model, destroy_qwen_moe_model, create_moe_kv_cache, drop_moe_kv_cache, infer_moe_batch, forward_moe_batch = initialize_moe_apis()

# 3. Import other local python modules
from infer_task import InferTask, KVCache
from tokenizers import decoders as _dec
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

# LlamaWeightsNaming can be reused as the per-layer non-MLP weights are named similarly
class LlamaWeightsNaming:
    def input_embd(self):
        return "model.embed_tokens.weight"

    def output_norm(self):
        return "model.norm.weight"

    def output_embd(self):
        return "lm_head.weight"

    def attn_norm(self, i):
        return f"model.layers.{i}.input_layernorm.weight"

    def attn_q(self, i):
        return f"model.layers.{i}.self_attn.q_proj.weight"

    def attn_k(self, i):
        return f"model.layers.{i}.self_attn.k_proj.weight"

    def attn_v(self, i):
        return f"model.layers.{i}.self_attn.v_proj.weight"

    def attn_o(self, i):
        return f"model.layers.{i}.self_attn.o_proj.weight"

    # MoE models typically don't have biases in attention
    def attn_q_b(self, i): return f"model.layers.{i}.self_attn.q_proj.bias"
    def attn_k_b(self, i): return f"model.layers.{i}.self_attn.k_proj.bias"
    def attn_v_b(self, i): return f"model.layers.{i}.self_attn.v_proj.bias"

    def attn_q_norm(self, i): return f"model.layers.{i}.self_attn.q_norm.weight"
    def attn_k_norm(self, i): return f"model.layers.{i}.self_attn.k_norm.weight"

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    # New MoE-specific naming conventions
    def moe_gate(self, i):
        return f"model.layers.{i}.mlp.gate.weight"

    def moe_expert_gate(self, i, j):
        return f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"

    def moe_expert_up(self, i, j):
        return f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"

    def moe_expert_down(self, i, j):
        return f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"

    def match(state_dict):
        return (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.weight" in state_dict
        )

# Specialized Meta loader for MoE models
class QwenMoeMetaFromConfig(QwenMoeMetaCStruct):
    def __init__(self, config, dtype=torch.bfloat16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_BF16

        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=config["num_key_value_heads"],
            dh=config["head_dim"],
            di=config["intermediate_size"], # This is for dense layers if any, can be ignored if all are sparse
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=config["rope_theta"],
            end_token=config["eos_token_id"],
            # New MoE fields
            num_experts=config["num_experts"],
            num_experts_per_tok=config["num_experts_per_tok"],
            moe_intermediate_size=config["moe_intermediate_size"],
            norm_topk_prob=1 if config.get("norm_topk_prob", False) else 0,
        )
        self.torch_dtype_logits = dtype

# Specialized and completely rewritten Weights loader for MoE models
class QwenMoeWeightsImpl(QwenMoeWeightsCStruct):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.bfloat16,
        torch_dt_norm=torch.float32,
        ndev=1,
        transpose_weight=True,
    ):
        # Most of the initial setup is the same
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        num_experts = meta.num_experts

        # Data type setup...
        if torch_dt_mat == torch.float16: self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32: self.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16: self.dt_mat = DataType.INFINI_DTYPE_BF16
        else: raise ValueError("Unsupported proj weight data type")
        if torch_dt_norm == torch.float16: self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32: self.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16: self.dt_norm = DataType.INFINI_DTYPE_BF16
        else: raise ValueError("Unsupported norm weight data type")

        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer

        # --- Global and Attention Weights (largely the same logic) ---
        # NOTE: MoE model has tie_word_embeddings=False, so we must load both.
        self.input_embd_tensor = state_dict[naming.input_embd()].to(meta.torch_dtype_logits)
        self.input_embd = self.input_embd_tensor.data_ptr()
        self.output_norm_tensor = state_dict[naming.output_norm()].to(torch_dt_norm)
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[naming.output_embd()].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(0, 1).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()
        
        # Attention weights... (This part is complex and model-specific, reusing a simplified version)
        self.attn_norm_tensors = [state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)]
        self.attn_norm_ptrs = [t.data_ptr() for t in self.attn_norm_tensors]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        # Simplified QKV loading for clarity
        def qkv_slices(_i):
            _Q = (
                state_dict[naming.attn_q(_i)]
                .reshape([nh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _K = (
                state_dict[naming.attn_k(_i)]
                .reshape([nkvh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _V = state_dict[naming.attn_v(_i)].reshape([nkvh, dh // 2, 2, d])
            _result = []
            _nh_per_dev = nh // ndev
            _nkvh_per_dev = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_Q[_idev * _nh_per_dev : (_idev + 1) * _nh_per_dev, :, :, :])
                _result.append(_K[_idev * _nkvh_per_dev : (_idev + 1) * _nkvh_per_dev, :, :, :])
                _result.append(_V[_idev * _nkvh_per_dev : (_idev + 1) * _nkvh_per_dev, :, :])
            return _result

        self.qkv_tensor = [
            torch.cat(qkv_slices(i), dim=0).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer): self.qkv_tensor[i] = self.qkv_tensor[i].transpose(0, 1).contiguous()
        self.qkv_tensor_ptrs = [t.data_ptr() for t in self.qkv_tensor]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)
        
        self.attn_o_tensor = [state_dict[naming.attn_o(i)].to(torch_dt_mat) for i in range(nlayer)]
        if not transpose_weight:
            for i in range(nlayer): self.attn_o_tensor[i] = self.attn_o_tensor[i].transpose(0, 1).contiguous()
        self.attn_o_ptrs = [t.data_ptr() for t in self.attn_o_tensor]
        self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)

        self.ffn_norm_tensors = [state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)]
        self.ffn_norm_ptrs = [t.data_ptr() for t in self.ffn_norm_tensors]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

        # --- MoE Weight Loading Logic (CORE NEW IMPLEMENTATION) ---
        self.moe_gate_tensors = []
        self.moe_experts_gate_up_tensors = []
        self.moe_experts_down_tensors = []

        print("Loading MoE weights...")
        for i in range(nlayer):
            # Load the gate for the current layer
            gate_tensor = state_dict[naming.moe_gate(i)].to(torch_dt_mat)
            self.moe_gate_tensors.append(gate_tensor)

            # Loop through all experts for the current layer
            for j in range(num_experts):
                gate_proj = state_dict[naming.moe_expert_gate(i, j)]
                up_proj = state_dict[naming.moe_expert_up(i, j)]
                down_proj = state_dict[naming.moe_expert_down(i, j)]

                # Combine gate and up projections, similar to dense FFNs
                gate_up_tensor = torch.cat([gate_proj, up_proj], dim=0).to(torch_dt_mat)
                
                # Append to the flattened lists
                self.moe_experts_gate_up_tensors.append(gate_up_tensor)
                self.moe_experts_down_tensors.append(down_proj.to(torch_dt_mat))
        
        print("Converting MoE weights to CTypes pointers...")
        # Convert Python lists of tensors to CTypes pointer arrays
        moe_gate_ptrs = [t.data_ptr() for t in self.moe_gate_tensors]
        self.moe_gate = (c_void_p * nlayer)(*moe_gate_ptrs)

        total_experts = nlayer * num_experts
        moe_experts_gate_up_ptrs = [t.data_ptr() for t in self.moe_experts_gate_up_tensors]
        self.moe_experts_gate_up = (c_void_p * total_experts)(*moe_experts_gate_up_ptrs)
        
        moe_experts_down_ptrs = [t.data_ptr() for t in self.moe_experts_down_tensors]
        self.moe_experts_down = (c_void_p * total_experts)(*moe_experts_down_ptrs)
        print("-" * 50)
        print(">>> Weight Loader Verification <<<")
        print(f"Expected layers (nlayer): {nlayer}")
        print(f"Expected experts per layer: {num_experts}")
        print(f"Total experts expected: {nlayer * num_experts}")
        print("-" * 50)
        print(f"Loaded gate tensors: {len(self.moe_gate_tensors)}")
        print(f"Loaded expert gate_up tensors: {len(self.moe_experts_gate_up_tensors)}")
        print(f"Loaded expert down tensors: {len(self.moe_experts_down_tensors)}")
        print("-" * 50)
        # 断言检查，如果数量不对，程序会直接报错
        assert len(self.moe_gate_tensors) == nlayer
        assert len(self.moe_experts_gate_up_tensors) == nlayer * num_experts
        assert len(self.moe_experts_down_tensors) == nlayer * num_experts
        print(">>> Verification PASSED: Correct number of MoE weights loaded.")
        print("-" * 50)

# BatchedTask can be reused if its structure is generic
class QwenMoeBatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (self.tokens, self.ntok, self.req_lens, self.nreq, self.req_pos,
                self.kv_caches, self.temperaturas, self.topks, self.topps)

# Main class for the MoE model
class QwenMoeForCausalLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        def load_all_safetensors_from_dir(dir_path_: str):
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            for file in sorted(dir_path_.glob("*.safetensors")):
                with safetensors.safe_open(file, "pt") as f:
                    for name_ in f.keys():
                        tensors_[name_] = f.get_tensor(name_)
            return tensors_

        print("Loading MoE model config and weights to host...")
        load_start_time = time.time()

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        
        # Assert that we are loading the correct model type
        assert "moe" in config.get("model_type", ""), "This script is for MoE models only."

        state_dict = load_all_safetensors_from_dir(model_dir_path)
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )
        
        self.meta = QwenMoeMetaFromConfig(config, max_tokens=max_tokens)
        self.weights = QwenMoeWeightsImpl(
            self.meta,
            LlamaWeightsNaming(),
            state_dict,
            ndev=ndev,
            transpose_weight=(device != DeviceType.DEVICE_TYPE_ASCEND),
        )

        load_end_time = time.time()
        print(f"Weight loading time: {load_end_time - load_start_time:.3f}s")

        print(f"Creating MoE model on {ndev} devices...")
        create_start_time = time.time()
        dev_ids = (c_int * ndev)(*range(ndev))
        
        self.model_instance = create_qwen_moe_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )
        create_end_time = time.time()
        print(f"Model creation time: {create_end_time - create_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return create_moe_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        drop_moe_kv_cache(self.model_instance, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = QwenMoeBatchedTask(tasks)
        infer_moe_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=0.95, topk_=20, temperature_=0.6):
        # Generation logic remains largely the same, just calling the new functions
        input_content_templated = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        print(input_content_templated, end="", flush=True)
        tokens = self.tokenizer.encode(input_content_templated)
        
        eos_token_id = self.config["eos_token_id"]
        eos_token_id_list = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id

        infer_task = InferTask(
            0, tokens, self.max_context_len(),
            temperature_, topk_, topp_, eos_token_id_list
        )
        infer_task.bind_kvcache(KVCache(self))
        
        output_content = ""
        for _ in range(max_steps):
            output_tokens = self.batch_infer_one_round([infer_task])
            if output_tokens[0] in eos_token_id_list:
                break
            
            output_str = self.tokenizer.decode(output_tokens[0])
            output_content += output_str
            print(output_str, end="", flush=True)
            
            infer_task.next(output_tokens[0])
        
        print("\n")
        infer_task._kv_cache.drop(self)
        return output_content

    def destroy_model_instance(self):
        destroy_qwen_moe_model(self.model_instance)
        print("MoE Model destroyed")

def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python qwen_moe.py [--cpu|--nvidia|...] <path/to/moe_model_dir> [n_device]"
        )
        sys.exit(1)
    
    model_path = sys.argv[2]
    device_map = {
        "--cpu": DeviceType.DEVICE_TYPE_CPU,
        "--nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
        "--cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
        "--ascend": DeviceType.DEVICE_TYPE_ASCEND,
        "--metax": DeviceType.DEVICE_TYPE_METAX,
        "--moore": DeviceType.DEVICE_TYPE_MOORE,
        "--iluvatar": DeviceType.DEVICE_TYPE_ILUVATAR,
    }
    device_type = device_map.get(sys.argv[1])
    if device_type is None:
        print(f"Invalid device type. Valid options: {list(device_map.keys())}")
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = QwenMoeForCausalLM(model_path, device_type, ndev)
    
    model.generate("你好，请介绍一下自己。", 100)
    
    model.destroy_model_instance()

if __name__ == "__main__":
    test()
