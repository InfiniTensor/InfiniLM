from ctypes import POINTER, c_int, c_uint, c_void_p, byref
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import asyncio

from libinfinicore_infer import (
    JiugeMeta,
    JiugeWeights,
    KVCache,
    DataType,
    DeviceType,
    create_jiuge_model,
    destroy_jiuge_model,
    create_kv_cache,
    drop_kv_cache,
    infer_batch,
)
import torch
import transformers

torch.set_default_device("cpu")

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

    def attn_q_b(self, i):
        return f"model.layers.{i}.self_attn.q_proj.bias"

    def attn_k_b(self, i):
        return f"model.layers.{i}.self_attn.k_proj.bias"

    def attn_v_b(self, i):
        return f"model.layers.{i}.self_attn.v_proj.bias"

    def ffn_norm(self, i):
        return f"model.layers.{i}.post_attention_layernorm.weight"

    def gate(self, i):
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"

    def match(state_dict):
        return (
            "model.norm.weight" in state_dict
            and "model.layers.0.self_attn.q_proj.weight" in state_dict
        )


class JiugeMetaFromLlama(JiugeMeta):
    def __init__(self, config, dtype=torch.float16):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        else:
            dt_ = DataType.INFINI_DTYPE_F16
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
            dh=config["hidden_size"] // config["num_attention_heads"],
            di=config["intermediate_size"],
            dctx=config["max_position_embeddings"],
            dvoc=config["vocab_size"],
            epsilon=config["rms_norm_eps"],
            theta=(config["rope_theta"] if "rope_theta" in config else 100000.0),
            end_token=2,
        )
        self.torch_dtype_logits = dtype


class JiugeWeightsImpl(JiugeWeights):
    def __init__(
        self,
        meta,
        naming,
        state_dict,
        torch_dt_mat=torch.float16,
        torch_dt_norm=torch.float32,
        ndev=1,
        transpose_weight=True,
    ):
        nlayer = meta.nlayer
        nh = meta.nh
        nkvh = meta.nkvh
        dh = meta.dh
        d = meta.d
        di = meta.di
        assert nh % nkvh == 0
        assert nh % ndev == 0
        assert nkvh % ndev == 0
        assert di % ndev == 0
        torch_dt_logits = meta.torch_dtype_logits
        if torch_dt_mat == torch.float16:
            self.dt_mat = DataType.INFINI_DTYPE_F16
        elif torch_dt_mat == torch.float32:
            self.dt_mat = DataType.INFINI_DTYPE_F32
        else:
            raise ValueError("Unsupported proj weight data type")
        if torch_dt_norm == torch.float16:
            self.dt_norm = DataType.INFINI_DTYPE_F16
        elif torch_dt_norm == torch.float32:
            self.dt_norm = DataType.INFINI_DTYPE_F32
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
        self.input_embd_tensor = state_dict[input_embd_naming].to(torch_dt_logits)
        self.input_embd = self.input_embd_tensor.data_ptr()
        self.output_norm_tensor = state_dict[naming.output_norm()].to(torch_dt_norm)
        self.output_norm = self.output_norm_tensor.data_ptr()
        self.output_embd_tensor = state_dict[output_embd_naming].to(torch_dt_mat)
        if not transpose_weight:
            self.output_embd_tensor = self.output_embd_tensor.transpose(0, 1).contiguous()
        self.output_embd = self.output_embd_tensor.data_ptr()

        self.attn_norm_tensors = [
            state_dict[naming.attn_norm(i)].to(torch_dt_norm) for i in range(nlayer)
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
            )
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
                self.qkv_tensor[i] = self.qkv_tensor[i].reshape(ndev, (nh + 2 * nkvh) // ndev * dh, d).transpose(1, 2).contiguous()
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

        self.attn_o_tensor = [
            state_dict[naming.attn_o(i)]
                .to(torch_dt_mat)
                .reshape([d, ndev, nh // ndev * dh])
                .transpose(0, 1)
                .contiguous()
            if transpose_weight 
            else state_dict[naming.attn_o(i)].transpose(0, 1).to(torch_dt_mat).contiguous()
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

        def gate_up_slices(_i):
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(state_dict[naming.gate(_i)][_start:_end, :])
                _result.append(state_dict[naming.up(_i)][_start:_end, :])
            return _result

        self.gate_up_tensors = [
            torch.concat(gate_up_slices(i)).to(torch_dt_mat) for i in range(nlayer)
        ]
        if not transpose_weight:
            for i in range(nlayer):
                self.gate_up_tensors[i] = self.gate_up_tensors[i].reshape(ndev, 2 * di // ndev, d).transpose(1, 2).contiguous()
        self.gate_up_ptrs = [self.gate_up_tensors[i].data_ptr() for i in range(nlayer)]
        self.ffn_gate_up = (c_void_p * nlayer)(*self.gate_up_ptrs)

        self.ffn_down_tensor = [
            state_dict[naming.down(i)]
            .to(torch_dt_mat)
            .reshape([d, ndev, di // ndev])
            .transpose(0, 1)
            .contiguous()
            if transpose_weight
            else state_dict[naming.down(i)].transpose(0, 1).to(torch_dt_mat).contiguous()
            for i in range(nlayer)
        ]
        self.ffn_down_ptrs = [self.ffn_down_tensor[i].data_ptr() for i in range(nlayer)]
        self.ffn_down = (c_void_p * nlayer)(*self.ffn_down_ptrs)


class JiugeForCauslLM:
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1):
        def load_all_safetensors_from_dir(dir_path_: str):
            tensors_ = {}
            dir_path_ = Path(dir_path_)
            for file in sorted(dir_path_.glob("*.safetensors")):
                data_ = safetensors.safe_open(file, "pt")
                for name_ in data_.keys():
                    tensors_[name_] = data_.get_tensor(name_)
            return tensors_
        
        print("Loading model weights to host...")
        load_start_time = time.time()

        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = [eos_token_id] if type(eos_token_id) == int else eos_token_id
        transpose_weight = device != DeviceType.DEVICE_TYPE_ASCEND # y = xW is faster than y=xW^T on Ascend
        if "llama" == config["model_type"]:
            model = transformers.LlamaForCausalLM.from_pretrained(model_dir_path).cpu().half()
            self.meta = JiugeMetaFromLlama(config)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)
            self.weights = JiugeWeightsImpl(
                self.meta, LlamaWeightsNaming(), model.state_dict(), ndev=ndev, transpose_weight=transpose_weight
            )
        elif "fm9g" == config["model_type"]:
            if any(file.suffix == ".safetensors" for file in Path(model_dir_path).iterdir()):
                state_dict = load_all_safetensors_from_dir(model_dir_path)
            else:
                state_dict = torch.load(
                os.path.join(model_dir_path, "pytorch_model.bin"), weights_only=True, map_location="cpu"
            )
            if LlamaWeightsNaming.match(state_dict):
                self.meta = JiugeMetaFromLlama(config)
                self.weights = JiugeWeightsImpl(
                    self.meta, LlamaWeightsNaming(), state_dict, ndev=ndev, transpose_weight=transpose_weight
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path, trust_remote_code=True
                )
            else:
                raise ValueError("Unsupported weight naming")
        elif "fm9g7b" == config["model_type"]:
            state_dict = torch.load(
                os.path.join(model_dir_path, "pytorch_model.bin"), weights_only=True, map_location="cpu"
            )
            if LlamaWeightsNaming.match(state_dict):
                self.meta = JiugeMetaFromLlama(config)
                self.weights = JiugeWeightsImpl(
                    self.meta, LlamaWeightsNaming(), state_dict, ndev=ndev, transpose_weight=transpose_weight
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path, trust_remote_code=True
                )
            else:
                raise ValueError("Unsupported weight naming")
        elif "qwen2" == config["model_type"]:
            state_dict = load_all_safetensors_from_dir(model_dir_path)
            if LlamaWeightsNaming.match(state_dict):
                self.meta = JiugeMetaFromLlama(config)
                self.weights = JiugeWeightsImpl(
                    self.meta, LlamaWeightsNaming(), state_dict, ndev=ndev, transpose_weight=transpose_weight
                )
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_dir_path
                )
        else:
            raise ValueError("Unsupported model architecture")

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")
        
        print(f"Creating model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.model_instance = create_jiuge_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")
        
    def create_kv_cache(self):
        return create_kv_cache(self.model_instance)

    def drop_kv_cache(self, kv_cache):
        drop_kv_cache(self.model_instance, kv_cache)

    def chat(self, request, kv_cache):
        messages = request.get("messages", [])
        temperature = request.get("temperature", 1.0)
        topk = request.get("top_k", 1)
        topp = request.get("top_p", 1.0)
        max_tokens = request.get("max_tokens", self.meta.dctx)
        input_content = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        tokens = self.tokenizer.encode(input_content)
        ntok = len(tokens)
        nreq = 1
        output_content = ""
        tokens = (c_uint * ntok)(*tokens)
        req_lens = (c_uint * nreq)(*[ntok])
        req_pos = (c_uint * nreq)(*[0])
        kv_caches = (POINTER(KVCache) * nreq)(*[kv_cache])
        ans = (c_uint * nreq)()

        steps = 0
        for step_i in range(max_tokens):
            infer_batch(
                self.model_instance,
                tokens,
                ntok,
                req_lens,
                nreq,
                req_pos,
                kv_caches,
                ans,
                temperature,
                topk,
                topp,
            )
            steps += 1
            output_tokens = list(ans)
            output_str = (
                self.tokenizer._tokenizer.id_to_token(output_tokens[0])
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
            )
            output_content += output_str

            if output_tokens[0] in self.eos_token_id:
                break
            req_pos[0] = req_pos[0] + ntok
            ntok = 1
            tokens = (c_uint * ntok)(*output_tokens)
            req_lens = (c_uint * nreq)(*[ntok])


        return output_content
    
    async def chat_stream_async(self, request, kv_cache):
        messages = request.get("messages", [])
        temperature = request.get("temperature", 1.0)
        topk = request.get("top_k", 1)
        topp = request.get("top_p", 1.0)
        max_tokens = request.get("max_tokens", 512)

        input_content = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        tokens = self.tokenizer.encode(input_content)
        ntok = len(tokens)
        nreq = 1
        tokens = (c_uint * ntok)(*tokens)
        req_lens = (c_uint * nreq)(*[ntok])
        req_pos = (c_uint * nreq)(*[0])
        kv_caches = (POINTER(KVCache) * nreq)(*[kv_cache])
        ans = (c_uint * nreq)()

        for step_i in range(max_tokens):
            infer_batch(
                self.model_instance,
                tokens,
                ntok,
                req_lens,
                nreq,
                req_pos,
                kv_caches,
                ans,
                temperature,
                topk,
                topp,
            )

            output_tokens = list(ans)
            output_str = (
                self.tokenizer._tokenizer.id_to_token(output_tokens[0])
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
            )

            yield output_str  # Yield each token as it's produced
            await asyncio.sleep(0)  # Let event loop breathe

            if output_tokens[0] in self.eos_token_id:
                break

            req_pos[0] += ntok
            ntok = 1
            tokens = (c_uint * ntok)(*output_tokens)
            req_lens = (c_uint * nreq)(*[ntok])

    def generate(self, input_content, max_steps, topp=1.0, topk=1, temperature=1.0):
        kv_cache = create_kv_cache(self.model_instance)
        input_content = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
        ntok = len(tokens)
        nreq = 1
        output_content = ""
        tokens = (c_uint * ntok)(*tokens)
        req_lens = (c_uint * nreq)(*[ntok])
        req_pos = (c_uint * nreq)(*[0])
        kv_caches = (POINTER(KVCache) * nreq)(*[kv_cache])
        ans = (c_uint * nreq)()

        steps = 0
        total_time = 0

        for step_i in range(max_steps):
            start_time = time.time()
            infer_batch(
                self.model_instance,
                tokens,
                ntok,
                req_lens,
                nreq,
                req_pos,
                kv_caches,
                ans,
                temperature,
                topk,
                topp,
            )
            steps += 1
            output_tokens = list(ans)
            end_time = time.time()
            output_str = (
                self.tokenizer._tokenizer.id_to_token(output_tokens[0])
                .replace("▁", " ")
                .replace("<0x0A>", "\n")
            )
            output_content += output_str
            print(output_str, end="", flush=True)
            if output_tokens[0] in self.eos_token_id:
                break
            req_pos[0] = req_pos[0] + ntok
            ntok = 1
            tokens = (c_uint * ntok)(*output_tokens)
            req_lens = (c_uint * nreq)(*[ntok])
            
            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1)
        print(f"Time per step: {avg_time:.3f}ms")
        for kv_cache in kv_caches:
            drop_kv_cache(self.model_instance, kv_cache)
        return output_content, avg_time
    
    def destroy_model_instance(self):
        destroy_jiuge_model(self.model_instance)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
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
    else:
        print(
            "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = JiugeForCauslLM(model_path, device_type, ndev)
    model.generate("山东最高的山是？", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()
