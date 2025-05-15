from ctypes import POINTER, c_uint, c_void_p, byref
import time
from libinfinicore_infer import (
    JiugeMeta,
    JiugeWeights,
    KVCache,
    DataType,
    DeviceType,
    create_jiuge_model,
    create_kv_cache,
    drop_kv_cache,
    infer_batch,
)
import torch
import transformers


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

    def gate(self, i):
        return f"model.layers.{i}.mlp.gate_proj.weight"

    def up(self, i):
        return f"model.layers.{i}.mlp.up_proj.weight"

    def down(self, i):
        return f"model.layers.{i}.mlp.down_proj.weight"


class JiugeMetaFromLlama(JiugeMeta):
    def __init__(self, config, infini_dtype):
        super().__init__(
            dt_logits=infini_dtype,
            dt_norm=infini_dtype,
            dt_mat=infini_dtype,
            nlayer=config.num_hidden_layers,
            d=config.hidden_size,
            nh=config.num_attention_heads,
            nkvh=(
                config.num_key_value_heads
                if config.num_key_value_heads
                else config.num_attention_heads
            ),
            dh=config.hidden_size // config.num_attention_heads,
            di=config.intermediate_size,
            dctx=config.max_position_embeddings,
            dvoc=config.vocab_size,
            epsilon=config.rms_norm_eps,
            theta=config.rope_theta,
            end_token=2,
        )


class JiugeWeightsImpl(JiugeWeights):
    def __init__(self, meta, naming, state_dict, ndev=1):
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
        self.input_embd = state_dict[naming.input_embd()].data_ptr()
        self.output_norm = state_dict[naming.output_norm()].data_ptr()
        self.output_embd = state_dict[naming.output_embd()].data_ptr()
        self.attn_norm = (c_void_p * nlayer)(
            *[state_dict[naming.attn_norm(i)].data_ptr() for i in range(nlayer)]
        )

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

        self.qkv_tensor = [torch.concat(qkv_slices(i)) for i in range(nlayer)]
        self.attn_qkv = (c_void_p * nlayer)(
            *[self.qkv_tensor[i].data_ptr() for i in range(nlayer)]
        )
        self.attn_o_tensor = [
            state_dict[naming.attn_o(i)]
            .reshape([d, ndev, nh // ndev * dh])
            .transpose(0, 1)
            .contiguous()
            for i in range(nlayer)
        ]
        self.attn_o = (c_void_p * nlayer)(
            *[self.attn_o_tensor[i].data_ptr() for i in range(nlayer)]
        )
        self.ffn_norm = (c_void_p * nlayer)(
            *[state_dict[naming.ffn_norm(i)].data_ptr() for i in range(nlayer)]
        )

        def gate_up_slices(_i):
            _result = []
            _di = di // ndev
            for _idev in range(ndev):
                _start = _idev * _di
                _end = (_idev + 1) * _di
                _result.append(state_dict[naming.gate(_i)][_start:_end, :])
                _result.append(state_dict[naming.up(_i)][_start:_end, :])
            return _result

        self.gate_up_tensor = [torch.concat(gate_up_slices(i)) for i in range(nlayer)]

        self.ffn_gate_up = (c_void_p * nlayer)(
            *[self.gate_up_tensor[i].data_ptr() for i in range(nlayer)]
        )

        self.ffn_down_tensor = [
            state_dict[naming.down(i)]
            .reshape([d, ndev, di // ndev])
            .transpose(0, 1)
            .contiguous()
            for i in range(nlayer)
        ]
        self.ffn_down = (c_void_p * nlayer)(
            *[self.ffn_down_tensor[i].data_ptr() for i in range(nlayer)]
        )


class JiugeForCauslLM:
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1):
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_dir_path, torch_dtype=torch.float16
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)
        self.meta = JiugeMetaFromLlama(model.config, DataType.INFINI_DTYPE_F16)
        self.weights = JiugeWeightsImpl(
            self.meta, LlamaWeightsNaming(), model.state_dict(), ndev=ndev
        )
        dev_ids = (c_uint * ndev)(*[i for i in range(ndev)])
        self.model_instance = create_jiuge_model(
            byref(self.meta),
            byref(self.weights),
            device,
            ndev,
            dev_ids,
        )

    def infer(self, input_list, topp=1.0, topk=1, temperature=1.0):
        pass

    def generate(self, input_content, max_steps, topp=1.0, topk=1, temperature=1.0):
        print(input_content, end="", flush=True)
        kv_cache = create_kv_cache(self.model_instance)
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
        start_time = time.time()
        for _ in range(max_steps):
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
            if output_str.endswith("</s>"):
                break
            output_content += output_str
            print(output_str, end="", flush=True)
            req_pos[0] = req_pos[0] + ntok
            ntok = 1
            tokens = (c_uint * ntok)(*output_tokens)
            req_lens = (c_uint * nreq)(*[ntok])

        print("\n")
        end_time = time.time()
        avg_time = (end_time - start_time) * 1000 / steps
        print(f"Time per step: {avg_time:.3f}ms")
        for kv_cache in kv_caches:
            drop_kv_cache(self.model_instance, kv_cache)
        return output_content, avg_time
