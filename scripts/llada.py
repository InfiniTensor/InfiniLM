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
    DeviceType,
    KVCacheCStruct,
    DataType
)
from libinfinicore_infer.llada import LLaDAModel, LLaDAMetaCStruct, LLaDAWeightsCStruct

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref

class LLaDAWeifghtsNaming:
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

    def attn_q_b(self, i):
        return f"model.layers.{i}.self_attn.q_proj.bias"

    def attn_k_b(self, i):
        return f"model.layers.{i}.self_attn.k_proj.bias"

    def attn_v_b(self, i):
        return f"model.layers.{i}.self_attn.v_proj.bias"

    def attn_q_norm(self, i):
        return f"model.layers.{i}.self_attn.q_norm.weight"

    def attn_k_norm(self, i):
        return f"model.layers.{i}.self_attn.k_norm.weight"

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def gate(self, i, j):
        return f"model.layers.{i}.mlp.expert.gate_proj.{j}.weight"

    def up(self, i, j):
        return f"model.layers.{i}.mlp.expert.up_proj.{j}.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"



class LLaDAMetaFromLlama(LLaDAMetaCStruct): # model specific data: heads num ....
    def __init__(self, config, dtype=torch.float16, max_tokens = None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16

        self.scale_input = 1.0
        self.scale_output = 1.0
        self.scale_o = 1.0
        self.scale_down = 1.0
        if (
            config["model_type"] in ["fm9g", "minicpm"]
            and "scale_emb" in config
            and "scale_depth" in config
            and "dim_model_base" in config
        ):
            self.scale_input = config["scale_emb"]
            self.scale_output = config["hidden_size"] //config["hidden_size"]
            self.scale_o = config["scale_depth"] / math.sqrt(
                config["num_hidden_layers"]
            )
            self.scale_down = config["scale_depth"] / math.sqrt(
                config["num_hidden_layers"]
            )

        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=(
                config["num_key_value_heads"]
                if "num_key_value_heads" in config
                else config["num_attention_heads"]
            ),
            dh=(
                config["head_dim"]
                if "head_dim" in config
                else config["hidden_size"] // config["num_attention_heads"]
            ),
            di_dense = config["dense_intermediate_size"],
            di_expert = config["expert_intermediate_size"],
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=(config["rope_theta"] if "rope_theta" in config else 100000.0),
            end_token=2,
            num_experts=config["num_experts"]
        )
        self.torch_dtype_logits = dtype


class LLaDAWeightsImpl(LLaDAWeightsCStruct):
    def __init__(self, meta, naming,
        state_dict,  # 权重
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float32,
        transpose_weight = None,
        ndev=1,
        ):
        nlayer = meta.nlayer
        nh = meta.nh
        di_expert = meta.di_expert
        di_dense = meta.di_dense
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        num_experts = meta.num_experts
        scale_input = meta.scale_input
        scale_output = meta.scale_output
        scale_o = meta.scale_o
        scale_down = meta.scale_down
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0

        torch_dt_logits = meta.torch_dtype_logits
        if torch_dt_mat == torch.float16:
            self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.dt_mat = DataType.INFINI_DTYPE_F32
        elif torch_dt_mat == torch.bfloat16:
            self.dt_mat = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported proj weight data type")
        if torch_dt_norm == torch.float16:
            self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.dt_norm = DataType.INFINI_DTYPE_F32
        elif torch_dt_norm == torch.bfloat16:
            self.dt_norm = DataType.INFINI_DTYPE_BF16
        else:
            raise ValueError("Unsupported norm weight data type")

        input_embd_naming = (
            naming.input_embd()
            if naming.input_embd() in state_dict
            else naming.output_embd()
        )
        output_embd_naming = (
            naming.output_embd()
            if naming.output_embd() in state_dict
            else naming.input_embd()
        )
        self.transpose_linear_weights = 1 if transpose_weight else 0
        self.nlayer = nlayer
        self.input_embd_tensor = (
            state_dict[input_embd_naming].to(torch_dt_logits) * scale_input
        )
        self.input_embd = self.input_embd_tensor.data_ptr()
        self.output_norm_tensor = (
            state_dict[naming.output_norm()].to(torch_dt_norm) * scale_output
        )
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(
                0, 1
            ).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer) # each layer's weight
        ]
        self.attn_norm_ptrs = [
            self.attn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.attn_norm = (c_void_p * nlayer)(*self.attn_norm_ptrs)

        def qkv_slices(_i):
            _Q = (
                state_dict[naming.attn_q(_i)]
                .reshape([nh, 2, dh // 2, d])
                .transpose(1, 2)
            ) # For RoPE
            _K = (
                state_dict[naming.attn_k(_i)]
                .reshape([nkvh, 2, dh // 2, d])
                .transpose(1, 2)
            )
            _V = state_dict[naming.attn_v(_i)].reshape([nkvh, dh // 2, 2, d])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_Q[_idev * _nh : (_idev + 1) * _nh, :, :, :])
                _result.append(_K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :, :])
                _result.append(_V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
            return _result

        self.qkv_tensor = [
            torch.concat(qkv_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.qkv_tensor[i] = (
                    self.qkv_tensor[i]
                    .reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d)
                    .transpose(1, 2)
                    .contiguous()
                )
        self.qkv_tensor_ptrs = [self.qkv_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)

        def qkv_b_slices(_i):
            _QB = (
                state_dict[naming.attn_q_b(_i)]
                .reshape([nh, 2, dh // 2])
                .transpose(1, 2)
            )
            _KB = (
                state_dict[naming.attn_k_b(_i)]
                .reshape([nkvh, 2, dh // 2])
                .transpose(1, 2)
            )
            _VB = state_dict[naming.attn_v_b(_i)].reshape([nkvh, dh // 2, 2])
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_QB[_idev * _nh : (_idev + 1) * _nh, :, :].flatten())
                _result.append(_KB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
                _result.append(_VB[_idev * _nkvh : (_idev + 1) * _nkvh, :, :].flatten())
            return _result

        if naming.attn_q_b(0) in state_dict:
            self.qkv_b_tensors = [
                torch.concat(qkv_b_slices(i)).to(torch_dt_logits) for i in range(nlayer)
            ]
            self.qkv_b_tensor_ptrs = [
                self.qkv_b_tensors[i].data_ptr() for i in range(nlayer)
            ]
            self.attn_qkv_b = (c_void_p * nlayer)(*self.qkv_b_tensor_ptrs)
        else:
            self.attn_qkv_b = None

        if naming.attn_q_norm(0) in state_dict:
            self.attn_q_norm_tensors = [
                state_dict[naming.attn_q_norm(i)]
                .reshape([2, dh // 2])
                .transpose(0, 1)
                .contiguous()
                .to(torch_dt_norm)
                for i in range(nlayer)
            ]
            self.attn_q_norm_ptrs = [
                self.attn_q_norm_tensors[i].data_ptr() for i in range(nlayer)
            ]
            self.attn_q_norm = (c_void_p * nlayer)(*self.attn_q_norm_ptrs)
            self.attn_k_norm_tensors = [
                state_dict[naming.attn_k_norm(i)]
                .reshape([2, dh // 2])
                .transpose(0, 1)
                .contiguous()
                .to(torch_dt_norm)
                for i in range(nlayer)
            ]
            self.attn_k_norm_ptrs = [
                self.attn_k_norm_tensors[i].data_ptr() for i in range(nlayer)
            ]
            self.attn_k_norm = (c_void_p * nlayer)(*self.attn_k_norm_ptrs)
        else:
            self.attn_q_norm = None
            self.attn_k_norm = None

        self.attn_o_tensor = [
            (
                state_dict[naming.attn_o(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, nh // ndev * dh])
                .transpose(0, 1)
                .contiguous()
                if transpose_weight
                else state_dict[naming.attn_o(i)]
                .transpose(0, 1)
                .to(torch_dt_mat)
                .contiguous()
            )
            * scale_o
            for i in range(nlayer)
        ]
        self.attn_o_ptrs = [self.attn_o_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_o = (c_void_p * nlayer)(*self.attn_o_ptrs)

        self.ffn_norm_tensors = [
            state_dict[naming.ffn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
        ]
        self.ffn_norm_ptrs = [
            self.ffn_norm_tensors[i].data_ptr() for i in range(nlayer)
        ]
        self.ffn_norm = (c_void_p * nlayer)(*self.ffn_norm_ptrs)

        def expert_gate_up_slices(layer_id, num_experts):
            """
            Extract expert gate and up weights for one layer.
            Compatible with keys like:
            model.layers.{i}.mlp.experts.{e}.gate_proj.weight
            model.layers.{i}.mlp.experts.{e}.up_proj.weight
            """
            gate_up_list = []

            for e in range(num_experts):
                gate_key = f"model.layers.{layer_id}.mlp.experts.{e}.gate_proj.weight"
                up_key   = f"model.layers.{layer_id}.mlp.experts.{e}.up_proj.weight"

                gate_w = state_dict[gate_key]     # shape: [1024, 2048]
                up_w   = state_dict[up_key]       # shape: [1024, 2048]

                # concat gate + up along dim 0 → shape: [2048, 2048]
                # this matches your previous behavior
                gate_up = torch.cat([gate_w, up_w], dim=0)

                gate_up_list.append(gate_up)

            return gate_up_list   # list of num_experts tensors
                
        self.gate_up_tensors = [
            torch.concat(expert_gate_up_slices(i, num_experts), dim=0).to(torch_dt_mat)
            for i in range(nlayer)
        ]
        # self.gate_up_tensors = [
        #     [t.to(torch_dt_mat) for t in expert_gate_up_slices(i, num_experts)]
        #     for i in range(nlayer)
        # ]
        # if not transpose_weight:
        #     for i in range(nlayer):
        #         self.gate_up_tensors[i] = (
        #             self.gate_up_tensors[i]
        #             .reshape(ndev, 2 * di_expert // ndev, d)
        #             .transpose(1, 2)
        #             .contiguous()
        #         ) #TODO: 具体切分设计
        self.gate_up_ptrs = [self.gate_up_tensors[i].data_ptr() for i in range(nlayer)]
        self.ffn_gate_up = (c_void_p * nlayer)(*self.gate_up_ptrs)
        

        def expert_down_slices(layer_id, num_experts):
            """
            Extract expert gate and up weights for one layer.
            Compatible with keys like:
            model.layers.{i}.mlp.experts.{e}.gate_proj.weight
            model.layers.{i}.mlp.experts.{e}.up_proj.weight
            """
            down_list = []

            for e in range(num_experts):
                down_key = f"model.layers.{layer_id}.mlp.experts.{e}.down_proj.weight"
                down_w = state_dict[down_key]     # shape: [1024, 2048]
                # concat gate + up along dim 0 → shape: [2048, 2048]
                # this matches your previous behavior
                down_list.append(down_w)

            return down_list   # list of num_experts tensors
        
        # self.ffn_down_tensor = [
        #     (
        #         state_dict[naming.down(i)]
        #         # .to(torch_dt_mat)
        #         # .reshape([d, ndev, di_expert // ndev])
        #         # .transpose(0, 1)
        #         # .contiguous()
        #         # if transpose_weight
        #         # else state_dict[naming.down(i)]
        #         # .transpose(0, 1)
        #         # .to(torch_dt_mat)
        #         # .contiguous() #TODO: 内存切分设计
        #     )
        #     * scale_down
        #     for i in range(nlayer)
        # ]
        self.ffn_down_tensor = [
            torch.concat(expert_down_slices(i, num_experts), dim=0).to(torch_dt_mat)
            for i in range(nlayer)
        ]
        self.ffn_down_ptrs = [self.ffn_down_tensor[i].data_ptr() for i in range(nlayer)]
        self.ffn_down = (c_void_p * nlayer)(*self.ffn_down_ptrs)

       

       




class LLaDAForCauslLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        def load_all_safetensors_from_dir(dir_path_: str): #TODO: Load. Accelerate By Page Cache
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            print(f"load Dir path {dir_path_}")
            for file in sorted(dir_path_.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, "pt")
                for name_ in data_.keys():
                    # print("Tensor Name ")
                    # print(name_)
                    tensors_[name_] = data_.get_tensor(name_)
            return tensors_
        
        print("Loading model weights to host...")
        load_start_time = time.time()

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )

        self.llada_model = LLaDAModel() # TODO: 实现LLaDAModel

        state_dict = load_all_safetensors_from_dir(model_dir_path)
        # C Structure Meta and weights
        self.meta = LLaDAMetaFromLlama(config, dtype=torch.bfloat16, max_tokens=max_tokens)
        self.weights = LLaDAWeightsImpl(
                    self.meta,
                    LLaDAWeifghtsNaming(),
                    state_dict,
                    ndev=ndev,
                    transpose_weight=None,
                    ) # bottleneck
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path, trust_remote_code=True
        ) # bottleneck
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")
        print(f"Creating model on {ndev} devices...")
        load_start_time = time.time()
        self.dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.ndev = ndev
        self.device = device
        print("--- start create model ---")
        # self.model_instance = self.llada_model.create_model()
        # self.model_instance = self.llada_model.create_model( # TODO:
        #     byref(self.meta),
        #     byref(self.weights),
        #     device,
        #     ndev,
        #     self.dev_ids,
        # )
        self.model_instance = self.llada_model.create_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            self.dev_ids,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")


def test():
    # if len(sys.argv) < 3:
    #     print(
    #         "Usage: python llada.py [--cpu | --nvidia] <path/to/model_dir> [n_device] [--verbose]"
    #     )
    #     sys.exit(1)

    # Parse command line arguments

    model_path = "/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/snapshots/783d3467f108d28ac0a78d3e41af16ab05cabd8d"
    device_type = DeviceType.DEVICE_TYPE_CPU
    verbose = False

    device_type = DeviceType.DEVICE_TYPE_CPU
        
    # Find n_device argument (skip --verbose)
    ndev = 1 # nums of card

    model = LLaDAForCauslLM(model_path, device_type, ndev)



if __name__ == "__main__":
    import os
    print(os.getpid())
    test()