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
from infer_task import InferTask, KVCache
from libinfinicore_infer import (
    DeviceType,
    KVCacheCStruct,
    DataType
)
from libinfinicore_infer.llada import LLaDAModel, LLaDAMetaCStruct, LLaDAWeightsCStruct
from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref
import torch.nn.functional as F
import numpy as np

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

    def router(self, i):
        return f"model.layers.{i}.mlp.gate.weight"

    def expert_gate(self, i, j):
        return f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"

    def expert_up(self, i, j):
        return f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"

    def down(self, i, j):
        return f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"



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
            _pad0 = 0,
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
            _pad1=0,
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
            state_dict[input_embd_naming].to(torch_dt_logits)
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
                # .reshape([nh, 2, dh // 2, d])
                # .transpose(1, 2)
            ) # For RoPE
            _K = (
                state_dict[naming.attn_k(_i)]
                # .reshape([nkvh, 2, dh // 2, d])
                # .transpose(1, 2)
            )
            # _V = state_dict[naming.attn_v(_i)].reshape([nkvh, dh // 2, 2, d])
            _V = state_dict[naming.attn_v(_i)]
            _result = []
            _nh = nh // ndev
            _nkvh = nkvh // ndev
            for _idev in range(ndev):
                _result.append(_Q)
                _result.append(_K)
                _result.append(_V)
            return _result

        self.qkv_tensor = [
            torch.concat(qkv_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        # if not transpose_weight:
        #     for i in range(nlayer):
        #         self.qkv_tensor[i] = (
        #             self.qkv_tensor[i]
        #             .reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d)
        #             .transpose(1, 2)
        #             .contiguous()
        #         )
        self.qkv_tensor_ptrs = [self.qkv_tensor[i].data_ptr() for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(*self.qkv_tensor_ptrs)

        
        if naming.attn_q_norm(0) in state_dict:
            print("have norm")
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

        def expert_gate_slices(layer_id, num_experts):
            """
            Extract expert gate and up weights for one layer.
            Compatible with keys like:
            model.layers.{i}.mlp.experts.{e}.gate_proj.weight
            model.layers.{i}.mlp.experts.{e}.up_proj.weight
            """
            gate_list = []

            for e in range(num_experts):
                gate_key = naming.expert_gate(layer_id, e)
                gate_w = state_dict[gate_key]     # shape: [1024, 2048]
                gate_list.append(gate_w)
            return gate_list   # list of num_experts tensors
        
        def expert_up_slices(layer_id, num_experts):
            """
            Extract expert gate and up weights for one layer.
            Compatible with keys like:
            model.layers.{i}.mlp.experts.{e}.gate_proj.weight
            model.layers.{i}.mlp.experts.{e}.up_proj.weight
            """
            up_list = []

            for e in range(num_experts):
                up_key = naming.expert_up
                up_key   = f"model.layers.{layer_id}.mlp.experts.{e}.up_proj.weight"
                up_w = state_dict[up_key]     # shape: [1024, 2048]
                up_list.append(up_w)
            return up_list   # list of num_experts tensors
        
        # memory: [gate_layer0_expert_gate0]...[gate_layer0_expert_gate63]......[gate_layer15_expert_gate63]
        self.expert_gate_tensors = [
            torch.concat(expert_gate_slices(i, num_experts), dim=0).to(torch_dt_mat)
            for i in range(nlayer)
        ]

        # memory: [gate_layer0_expert_up0]...[gate_layer0_expert_up63]......[gate_layer15_expert_up63]
        self.expert_up_tensors = [
            torch.concat(expert_up_slices(i, num_experts), dim=0).to(torch_dt_mat)
            for i in range(nlayer)
        ]

        self.expert_gate_ptrs = [self.expert_gate_tensors[i].data_ptr() for i in range(nlayer)]
        self.expert_gate = (c_void_p * nlayer)(*self.expert_gate_ptrs)
        
        self.expert_up_ptrs = [self.expert_up_tensors[i].data_ptr() for i in range(nlayer)]
        self.expert_up    = (c_void_p * nlayer)(*self.expert_up_ptrs)

        def expert_down_slices(layer_id, num_experts):
            """
            Extract expert gate and up weights for one layer.
            Compatible with keys like:
            model.layers.{i}.mlp.experts.{e}.gate_proj.weight
            model.layers.{i}.mlp.experts.{e}.up_proj.weight
            """
            down_list = []

            for e in range(num_experts):
                down_key = naming.down(layer_id, e)
                down_w = state_dict[down_key]     # shape: [1024, 2048]
                # concat gate + up along dim 0 → shape: [2048, 2048]
                down_list.append(down_w)
            return down_list   # list of num_experts tensors
        
        # memory: [gate_layer0_expert_down0]...[gate_layer0_expert_down63]......[gate_layer15_expert_down63]
        self.expert_down_tensor = [
            torch.concat(expert_down_slices(i, num_experts), dim=0).to(torch_dt_mat)
            for i in range(nlayer)
        ]
        self.expert_down_ptrs = [self.expert_down_tensor[i].data_ptr() for i in range(nlayer)]
        self.expert_down = (c_void_p * nlayer)(*self.expert_down_ptrs)

        # Impl Python gate 
        def router_slices():
            """
            Extract expert gate and up weights for one layer.
            Compatible with keys like:
            model.layers.{i}.mlp.experts.{e}.gate_proj.weight
            model.layers.{i}.mlp.experts.{e}.up_proj.weight
            """
            router_list = []
            for i in range(nlayer):
                gate_weight = state_dict[naming.router(i)].to(torch_dt_mat)
                
                router_list.append(gate_weight)
            
            return router_list   # list of num_experts tensors
        
        self.router_gate_tensor = router_slices()
        # memory: [gate_layer0_router]......[gate_layer15_router]
        self.router_ptrs = [self.router_gate_tensor[i].data_ptr() for i in range(nlayer)]
        self.router = (c_void_p * nlayer)(*self.router_ptrs)
        


class LLaDABatchedTask:
    """
    Batch task handler for LLaDA model inference.
    Similar to JiugeBatchedTask but adapted for LLaDA requirements.
    """
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



        flat_tokens = []
        for toks in token_lists:
            if isinstance(toks, (list, tuple)):
                flat_tokens.extend(toks)
            else:
                flat_tokens.append(toks)

        # Convert all tokens to int
        flat_tokens = [int(tok) for tok in flat_tokens]

        self.ntok = len(flat_tokens)
        print(f"Torch : flat_tokens : {flat_tokens}")

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
        self.model_instance = self.llada_model.create_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            self.dev_ids,
        )
        self.model_ptr = self.model_instance
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")


    # <--------------------------------------  Infer PipeLine  ------------------------------------------------>
    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        """Create KV cache for the model"""
        return self.llada_model.create_kv_cache(
            self.meta.nlayer,
            self.meta.dctx,
            self.meta.nkvh,
            self.meta.dh,
            self.meta.dh,
            self.meta.dt_logits,
            self.device,
            self.dev_ids,
            self.ndev,
        )

    def drop_kv_cache(self, kv_cache):
        """Drop KV cache"""
        self.llada_model.drop_kv_cache(kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        """
        Perform one round of batch inference using LLaDA model.

        Args:
            tasks: List of InferTask objects containing input sequences and parameters

        Returns:
            List of generated token IDs
        """
        output = (c_uint * len(tasks))()
        batch_inputs = LLaDABatchedTask(tasks)
        self.llada_model.infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)
    
    def _sample_next_token(self, logits: torch.Tensor, temperature: float = 0.0, topk: int = 1, topp: float = 1.0) -> int:
        """
        Sample the next token from logits.

        Args:
            logits: Tensor of shape (vocab_size,)
            temperature: Sampling temperature (0 = greedy)
            topk: Number of top tokens to consider
            topp: Cumulative probability threshold for nucleus sampling

        Returns:
            Sampled token ID
        """
        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

        # Apply softmax
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)

        # Top-k filtering
        if topk > 1:
            topk_values, topk_indices = torch.topk(probs, k=topk, dim=-1)
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, topk_indices, topk_values)
            # Renormalize
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # Top-p (nucleus) filtering
        if topp < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Create mask for tokens within top-p
            sorted_mask = cumulative_probs <= topp
            # Apply mask and renormalize
            filtered_probs = sorted_probs * sorted_mask.float()
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            # Scatter back to original order
            probs = torch.zeros_like(probs)
            probs.scatter_(-1, sorted_indices, filtered_probs)

        # Sample
        if temperature > 0:
            # Multinomial sampling
            probs_flat = probs.view(-1)
            next_token = torch.multinomial(probs_flat, num_samples=1).item()
        else:
            # Greedy sampling (argmax)
            next_token = torch.argmax(probs, dim=-1).item()

        return next_token

    def generate(
        self,
        prompts: str,
        max_steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature_: float = 0.,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence',
        mask_id: int = 126336,
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
        verbose: bool = False,
        topp_ = 1.0,
        topk_ = 1
    ):
        # if isinstance(prompts, str):
        #     prompts = [prompts]
        # messages = [
        #     {"role": "system", "content": "You are a helpful AI assistant."},
        #     {"role": "user", "content": prompts[0]} 
        # ]
        # # messages = [{"role": "user", "content": prompt} for prompt in prompts]
        # formatted_prompts = [self.tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]
        # encoded_outputs = self.tokenizer.batch_encode_plus(
        #     formatted_prompts,
        #     add_special_tokens=False,
        #     padding=True,
        #     return_tensors="pt"
        # )

        all_messages = []
       
        messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompts}
        ]

        # 对每个完整对话应用模板
        formatted_prompts = [
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ]
        
        # 批量编码
        encoded_outputs = self.tokenizer(formatted_prompts,)

        # Extract input_ids from batch encoding
        input_ids = encoded_outputs['input_ids']

        # For single prompt, get the first sequence

        tokens = input_ids

        for i in range(128):
            tokens[0].append(156895)

        print(len(tokens))
        print(f"Pytho Side Tokens type: {type(tokens)}, content: {tokens if isinstance(tokens, list) else 'Not a list'}")

        # Prepare vocab size for logits tensor
        vocab_size = self.config.get("vocab_size", 150528)

        # Create KV cache
        kv_cache = KVCache(self)

        # Auto-regressive generation
        generated_tokens = tokens.copy() if isinstance(tokens, list) else tokens[0].copy()
        print("Staring Auto-regressive Generation")

        for step in range(gen_length):
            # Check for EOS token
            if generated_tokens[-1] in self.eos_token_id:
                print(f"EOS token reached at step {step}")
                break

            # Prepare logits tensor - pass the full sequence each time
            # LLaDA uses bidirectional attention, not causal autoregressive
            batch_tokens_tensor = torch.tensor(generated_tokens, dtype=torch.long, device="cpu")

            # Allocate memory for logits output from C++
            seq_len = len(generated_tokens[0])
            logits_np = np.zeros((seq_len, vocab_size), dtype=np.float32)
            logits_ptr = logits_np.ctypes.data_as(POINTER(c_float))

            # Call C++ forward_batch to get logits
            # For bidirectional models like LLaDA, req_pos is typically 0 (process full sequence)
            self.llada_model.forward_batch(
                self.model_instance,
                batch_tokens_tensor.numpy().astype(np.uint32).ctypes.data_as(POINTER(c_uint)), 
                seq_len,
                (c_uint * 1)(seq_len),
                1,  # nreq
                (c_uint * 1)(0),  # req_pos = 0 for bidirectional attention (process full sequence)
                kv_cache.data(),
                logits_ptr,  # Pass the logits buffer to C++
            )

            # Convert to torch tensor
            logits_tensor = torch.from_numpy(logits_np).float()

            # Sample next token from the last position's logits
            # logits_tensor shape: [seq_len, vocab_size], we need last row
            last_token_logits = logits_tensor[-1, :]  # [vocab_size]
            next_token = self._sample_next_token(last_token_logits, temperature_, topk_, topp_)

            # Append to generated tokens
            generated_tokens.append(next_token)

            if verbose and step % 10 == 0:
                print(f"Step {step}: Generated token {next_token}, Sequence length: {len(generated_tokens)}")

        # # Clean up KV cache
        # kv_cache.drop()

        # Decode the generated tokens
        generated_text = self.tokenizer.decode(generated_tokens[len(tokens):], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

        return generated_tokens, generated_text





    def forward_logits_batch(self, input_ids_tensor, attention_mask_tensor=None):
        """
        Forward pass to get logits for a batch of sequences using C++ model.

        Args:
            input_ids_tensor: Tensor of shape (batch_size, seq_len) with token IDs
            attention_mask_tensor: Tensor of shape (batch_size, seq_len) with attention mask

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids_tensor.shape

        # Create InferTask objects for each sequence in the batch
        tasks = []
        for i in range(batch_size):
            # Extract tokens for this sequence
            seq_tokens = input_ids_tensor[i].tolist()
            # Create KVCache for this sequence
            kv_cache = KVCache(
                self.meta.nlayer,
                self.meta.dctx,
                self.meta.nkvh,
                self.meta.dh,
                self.meta.dh,
                self.meta.dt_logits,
                self.device,
                self.dev_ids,
                self.ndev,
            )
            # Create InferTask
            task = InferTask(
                tokens=seq_tokens,
                pos=0,  # Start position
                temperature=0.0,  # Will be handled by sampling logic
                topk=1,  # Will be handled by sampling logic
                topp=1.0,  # Will be handled by sampling logic
                kvcache=kv_cache,
            )
            tasks.append(task)

        # Create batched task
        batch_inputs = LLaDABatchedTask(tasks)

        # Prepare output tensor for logits
        vocab_size = self.config.get("vocab_size", 150528)
        logits_tensor = torch.zeros(
            batch_inputs.ntok, vocab_size,
            dtype=self.meta.torch_dtype_logits,
            device=torch.device("cpu")
        )

        # Call C++ forward_batch
        self.llada_model.forward_batch(
            self.model_instance,
            batch_inputs.tokens,
            batch_inputs.ntok,
            batch_inputs.req_lens,
            batch_inputs.nreq,
            batch_inputs.req_pos,
            batch_inputs.kv_caches,
            logits_tensor.data_ptr(),
        )

        # Reshape logits to (batch_size, seq_len, vocab_size)
        # Note: This requires careful handling of the flattened output
        logits_reshaped = torch.zeros(batch_size, seq_len, vocab_size, dtype=logits_tensor.dtype, device=logits_tensor.device)

        # Copy logits back to batch format
        token_offset = 0
        for req_idx, req_len in enumerate(batch_inputs.req_lens_list):
            # Extract logits for this request
            req_logits = logits_tensor[token_offset:token_offset + req_len]
            logits_reshaped[req_idx, :req_len] = req_logits
            token_offset += req_len

        # Clean up KV caches
        for task in tasks:
            task.kvcache().drop()

        return logits_reshaped




def test():
    model_path = "/home/featurize/work/InfiniFamily/cache/models--inclusionAI--LLaDA-MoE-7B-A1B-Instruct/snapshots/783d3467f108d28ac0a78d3e41af16ab05cabd8d"
    device_type = DeviceType.DEVICE_TYPE_NVIDIA
    verbose = True

    # Number of devices
    ndev = 1

    print("Loading LLaDA model...")
    model = LLaDAForCauslLM(model_path, device_type, ndev)

    # # Load PyTorch original model for comparison
    # print("\n=== Loading PyTorch Model for Comparison ===")
    # from transformers import AutoModelForCausalLM
    # import json
    # torch_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map={"": 0} )
    # torch_model.eval()

    # Test prompts
    test_prompts = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    print("\n=== Testing C++ Model ===")
    cpp_result = model.generate(
                prompts=test_prompts,
                max_steps=16,  # Reduced for faster testing
                gen_length=32,  # Shorter generation for testing
                block_length=16,
                temperature_=0.0,  # Deterministic for testing
                verbose=verbose
            )

    # # Test with PyTorch model
    # print("\n=== Testing PyTorch Model ===")
    # with torch.no_grad():
    #     messages = [{"role": "user", "content": test_prompts}]
    #     formatted_prompts = [model.tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]
    #     encoded_inputs = model.tokenizer.batch_encode_plus(
    #         formatted_prompts,
    #         add_special_tokens=False,
    #         padding=True,
    #         return_tensors="pt"
    #     )
    #     input_ids = encoded_inputs['input_ids']
    #     if len(test_prompts) == 1:
    #         tokens = input_ids[0].tolist()
    #     else:
    #         tokens = [seq.tolist() for seq in input_ids]

    #     # Generate with PyTorch
    #     with torch.no_grad():
    #         # Forward pass through all layers
    #         pytorch_outputs = torch_model(
    #             input_ids=input_ids,
    #             attention_mask=None,
    #             position_ids=None,
    #             output_router_logits=False,
    #             output_hidden_states=False,
    #             use_cache=False,
    #             return_dict=True,
    #         )

    #     # Get logits from PyTorch output
    #     pytorch_logits = pytorch_outputs.logits
    #     pytorch_logits = pytorch_logits.to('cpu')

    #     # Generate tokens using same sampling logic
    #     generated_tokens_pytorch = tokens.copy() if isinstance(tokens, list) else tokens[0].copy()
    #     print(f"PyTorch tokens: {len(generated_tokens_pytorch)}, content: {generated_tokens_pytorch}")

    #     for step in range(32):
    #         if generated_tokens_pytorch[-1] in model.eos_token_id:
    #             print(f"PyTorch EOS reached at step {step}")
    #             break

    #         # Get last token logits
    #         last_token_logits = pytorch_logits[:, -1, :]  # [vocab_size]
    #         next_token = model._sample_next_token(last_token_logits, 0.0, 1, 1.0)
    #         generated_tokens_pytorch.append(next_token)

    #         if verbose and step % 10 == 0:
    #             print(f"PyTorch Step {step}: Generated token {next_token}, Sequence length: {len(generated_tokens_pytorch)}")

    #     pytorch_generated_text = model.tokenizer.decode(generated_tokens_pytorch[len(tokens):], skip_special_tokens=True)
    #     print(f"PyTorch Generated text: {pytorch_generated_text}")

    # # Compare results
    # print("\n=== Comparison Results ===")
    # print(f"C++ Generated tokens: {cpp_result[0]}")
    # print(f"PyTorch Generated tokens: {generated_tokens_pytorch}")
    # print(f"Tokens match: {cpp_result[0] == generated_tokens_pytorch}")

    # # Decode for display
    # cpp_text = model.tokenizer.decode(cpp_result[0], skip_special_tokens=True)
    # pytorch_text = model.tokenizer.decode(generated_tokens_pytorch, skip_special_tokens=True)
    # print(f"\nC++ Output:\n{cpp_text}")
    # print(f"\nPyTorch Output:\n{pytorch_text}")

if __name__ == "__main__":
    import os
    print(os.getpid())
    test()