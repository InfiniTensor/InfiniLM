import ctypes
from typing import List, Sequence

from tqdm import tqdm

from libinfinicore_infer import (
    Qwen3vlModel,
    Qwen3vlMetaCStruct,
    TextMetaCStruct,
    VisMetaCStruct,
    Qwen3vlWeightsCStruct,
    Qwen3vlCacheCStruct,
    DataType,
    DeviceType,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_uint16, c_void_p, byref, c_bool
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


class Qwen3vlLangWeightsNaming:
    def input_embd(self):
        return "model.language_model.embed_tokens.weight"
    
    def output_embd(self):
        return "model.language_model.embed_tokens.weight"
    
    def output_norm(self):
        return "model.language_model.norm.weight"
    
    def attn_norm(self, i):
        return f"model.language_model.layers.{i}.input_layernorm.weight"
    
    def attn_q_proj(self, i):
        return f"model.language_model.layers.{i}.self_attn.q_proj.weight"
    
    def attn_q_norm(self, i):
        return f"model.language_model.layers.{i}.self_attn.q_norm.weight"
    
    def attn_k_proj(self, i):
        return f"model.language_model.layers.{i}.self_attn.k_proj.weight"
    
    def attn_k_norm(self, i):
        return f"model.language_model.layers.{i}.self_attn.k_norm.weight"
    
    def attn_o_proj(self, i):
        return f"model.language_model.layers.{i}.self_attn.o_proj.weight"
    
    def attn_v_proj(self, i):
        return f"model.language_model.layers.{i}.self_attn.v_proj.weight"
    
    def mlp_norm(self, i):
        return f"model.language_model.layers.{i}.post_attention_layernorm.weight"
    
    def mlp_gate(self, i):
        return f"model.language_model.layers.{i}.mlp.gate_proj.weight"
    
    def mlp_down(self, i):
        return f"model.language_model.layers.{i}.mlp.down_proj.weight"
    
    def mlp_up(self, i):
        return f"model.language_model.layers.{i}.mlp.up_proj.weight"
    
class Qwen3vlVisWeightsNaming:
    def patch_embed_weight(self):
        return "model.visual.patch_embed.proj.weight"
    def patch_embed_bias(self):
        return "model.visual.patch_embed.proj.bias"
    def pos_embed_weight(self):
        return "model.visual.pos_embed.weight"
    def attn_proj_weight(self,i):
        return f"model.visual.blocks.{i}.attn.proj.weight"
    def attn_proj_bias(self,i):
        return f"model.visual.blocks.{i}.attn.proj.bias"
    def attn_qkv_weight(self,i):
        return f"model.visual.blocks.{i}.attn.qkv.weight"
    def attn_qkv_bias(self,i):
        return f"model.visual.blocks.{i}.attn.qkv.bias"
    def mlp_linear_fc1_weight(self,i):
        return f"model.visual.blocks.{i}.mlp.linear_fc1.weight"
    def mlp_linear_fc1_bias(self,i):
        return f"model.visual.blocks.{i}.mlp.linear_fc1.bias"
    def mlp_linear_fc2_weight(self,i):
        return f"model.visual.blocks.{i}.mlp.linear_fc2.weight"
    def mlp_linear_fc2_bias(self,i):
        return f"model.visual.blocks.{i}.mlp.linear_fc2.bias"
    def norm1_weight(self,i):
        return f"model.visual.blocks.{i}.norm1.weight"
    def norm1_bias(self,i):
        return f"model.visual.blocks.{i}.norm1.bias"
    def norm2_weight(self,i):
        return f"model.visual.blocks.{i}.norm2.weight"
    def norm2_bias(self,i):
        return f"model.visual.blocks.{i}.norm2.bias"
    def deepstack_merger_linear_fc1_weight(self,i):
        return f"model.visual.deepstack_merger_list.{i}.linear_fc1.weight"
    def deepstack_merger_linear_fc1_bias(self,i):
        return f"model.visual.deepstack_merger_list.{i}.linear_fc1.bias"
    def deepstack_merger_linear_fc2_weight(self,i):
        return f"model.visual.deepstack_merger_list.{i}.linear_fc2.weight"
    def deepstack_merger_linear_fc2_bias(self,i):
        return f"model.visual.deepstack_merger_list.{i}.linear_fc2.bias"
    def deepstack_merger_norm_weight(self,i):
        return f"model.visual.deepstack_merger_list.{i}.norm.weight"
    def deepstack_merger_norm_bias(self,i):
        return f"model.visual.deepstack_merger_list.{i}.norm.bias"
    
    def merger_linear_fc1_weight(self):
        return "model.visual.merger.linear_fc1.weight"
    def merger_linear_fc1_bias(self):
        return "model.visual.merger.linear_fc1.bias"
    def merger_linear_fc2_weight(self):
        return "model.visual.merger.linear_fc2.weight"
    def merger_linear_fc2_bias(self):
        return "model.visual.merger.linear_fc2.bias"
    def merger_norm_weight(self):
        return "model.visual.merger.norm.weight"
    def merger_norm_bias(self):
        return "model.visual.merger.norm.bias"
    
class Qwen3vlMeta(Qwen3vlMetaCStruct):
    def __init__(self, config, max_tokens=None):

        if config['text_config']['dtype'] == 'float16':
            dt_ = DataType.INFINI_DTYPE_F16
            self.torch_dtype = torch.float16
        elif config['text_config']['dtype'] == 'float32':
            dt_ = DataType.INFINI_DTYPE_F32
            self.torch_dtype = torch.float32
        elif config['text_config']['dtype'] == 'bfloat16':
            dt_ = DataType.INFINI_DTYPE_BF16
            self.torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported text dtype: {config['text_config']['dtype']}")

        super().__init__(
            dtype = dt_,
            image_token_id = config['image_token_id'],
            video_token_id = config['video_token_id'],
            vision_end_token_id = config['vision_end_token_id'],
            vision_start_token_id = config['vision_start_token_id'],
            text_meta = TextMetaCStruct(
                bos_token_id = config['text_config']['bos_token_id'],
                eos_token_id = config['text_config']['eos_token_id'],
                head_dim = config['text_config']['head_dim'],
                hidden_size = config['text_config']['hidden_size'],
                initializer_range = config['text_config']['initializer_range'],
                intermediate_size = config['text_config']['intermediate_size'],
                max_tokens = (config['text_config']['max_position_embeddings'] if max_tokens is None else max_tokens),
                num_attention_heads = config['text_config']['num_attention_heads'],
                num_hidden_layers = config['text_config']['num_hidden_layers'],
                num_key_value_heads = config['text_config']['num_key_value_heads'],
                rms_norm_eps = config['text_config']['rms_norm_eps'],
                mrope_section = (ctypes.c_ulong * 3)(*config['text_config']['rope_scaling']['mrope_section']),
                rope_theta = config['text_config']['rope_theta'],
                vocab_size = config['text_config']['vocab_size'],
            ),
            vis_meta = VisMetaCStruct(
                depth = config['vision_config']['depth'],
                deepstack_visual_indexes = (ctypes.c_ulong * 3)(*config['vision_config']['deepstack_visual_indexes']),
                hidden_size = config['vision_config']['hidden_size'],
                in_channels = config['vision_config']['in_channels'],
                initializer_range = config['vision_config']['initializer_range'],
                intermediate_size = config['vision_config']['intermediate_size'],
                num_heads = config['vision_config']['num_heads'],
                num_position_embeddings = config['vision_config']['num_position_embeddings'],
                out_hidden_size = config['vision_config']['out_hidden_size'],
                patch_size = config['vision_config']['patch_size'],
                spatial_merge_size = config['vision_config']['spatial_merge_size'],
                temporal_patch_size = config['vision_config']['temporal_patch_size']
            )
        )

def load_specific_tensor(model_dir, tensor_name):
    """
    Load a specific tensor from a safetensors model.
    Supports both sharded models (with index.json) and single file models.
    """
    
    # Try to load from individual .safetensors files
    safetensors_files = [f for f in os.listdir(model_dir) if f.endswith(".safetensors")]
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    # Try to find the tensor in each file
    for filename in safetensors_files:
        tensor_file = os.path.join(model_dir, filename)
        try:
            with safetensors.safe_open(tensor_file, framework="pt", device="cpu") as f:
                if tensor_name in f.keys():
                    tensor = f.get_tensor(tensor_name)
                    return tensor
        except Exception:
            continue
            
    # If we reach here, tensor was not found in any file
    raise KeyError(f"{tensor_name} not found in any .safetensors files")

def load_Qwen3vl_weights(
    meta: Qwen3vlMeta,
    weights,
    model_path: str,
    ndev: int,
):
    # torch load weights, and reshape for qkv_proj / mlp_gate_up stack, attn / mlp parallel
    # weight loader function load from specific offset according to idev, and transpose
    model_instance = Qwen3vlModel()
    weight_loader = model_instance.create_weight_loader()
    vis_names = Qwen3vlVisWeightsNaming()
    lang_names = Qwen3vlLangWeightsNaming()

    nkvh = meta.text_meta.num_key_value_heads
    nh = meta.text_meta.num_attention_heads
    dh = meta.text_meta.head_dim
    d = meta.text_meta.hidden_size
    di = meta.text_meta.intermediate_size

    assert nh % nkvh == 0
    assert nh % ndev == 0
    assert nkvh % ndev == 0
    assert di % ndev == 0

    # -------------------------------
    # Language_model weights
    # -------------------------------
    input_embd = load_specific_tensor(model_path, lang_names.input_embd()).to(meta.torch_dtype)
    weight_loader.contents.lang_loader.load_input_embd(weights, input_embd.data_ptr())
    del input_embd

    output_norm = load_specific_tensor(model_path, lang_names.output_norm()).to(meta.torch_dtype)
    weight_loader.contents.lang_loader.load_output_norm(weights, output_norm.data_ptr())
    del output_norm

    output_embd = load_specific_tensor(model_path, lang_names.output_embd()).to(meta.torch_dtype)
    weight_loader.contents.lang_loader.load_output_embd(weights, output_embd.data_ptr())
    del output_embd

    for i in range(meta.text_meta.num_hidden_layers):
        attn_norm = load_specific_tensor(model_path, lang_names.attn_norm(i)).to(meta.torch_dtype)
        weight_loader.contents.lang_loader.load_attn_norm(weights, attn_norm.data_ptr(), i)
        del attn_norm

        attn_q_proj = load_specific_tensor(model_path, lang_names.attn_q_proj(i))
        attn_k_proj = load_specific_tensor(model_path, lang_names.attn_k_proj(i))
        attn_v_proj = load_specific_tensor(model_path, lang_names.attn_v_proj(i))
        
        _Q = attn_q_proj.reshape(nh,dh,d)
        _K = attn_k_proj.reshape(nkvh,dh,d)
        _V = attn_v_proj.reshape(nkvh,dh,d)

        qkv_proj = []
        _nh = nh // ndev
        _nkvh = nkvh // ndev
        for _idev in range(ndev):
            qkv_proj.append(_Q[_idev * _nh : (_idev + 1) * _nh, :, :])
            qkv_proj.append(_K[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
            qkv_proj.append(_V[_idev * _nkvh : (_idev + 1) * _nkvh, :, :])
        attn_qkv_proj = torch.cat(qkv_proj, dim=0).to(meta.torch_dtype).contiguous()

        weight_loader.contents.lang_loader.load_attn_qkv_proj(weights, attn_qkv_proj.data_ptr(), i)
        del attn_qkv_proj

        attn_q_norm = load_specific_tensor(model_path, lang_names.attn_q_norm(i)).to(meta.torch_dtype)
        weight_loader.contents.lang_loader.load_attn_q_norm(weights, attn_q_norm.data_ptr(), i)
        del attn_q_norm

        attn_k_norm = load_specific_tensor(model_path, lang_names.attn_k_norm(i)).to(meta.torch_dtype)
        weight_loader.contents.lang_loader.load_attn_k_norm(weights, attn_k_norm.data_ptr(), i)
        del attn_k_norm

        attn_o_proj = load_specific_tensor(model_path, lang_names.attn_o_proj(i))
        attn_o_proj = attn_o_proj.to(meta.torch_dtype).reshape([d, ndev, nh // ndev * dh]).transpose(0, 1).contiguous()
        weight_loader.contents.lang_loader.load_attn_o_proj(weights, attn_o_proj.data_ptr(), i)
        del attn_o_proj

        mlp_norm = load_specific_tensor(model_path, lang_names.mlp_norm(i)).to(meta.torch_dtype)
        weight_loader.contents.lang_loader.load_mlp_norm(weights, mlp_norm.data_ptr(), i)
        del mlp_norm

        mlp_gate = load_specific_tensor(model_path, lang_names.mlp_gate(i))
        mlp_up = load_specific_tensor(model_path, lang_names.mlp_up(i))
        
        gate_up = []
        _di = di // ndev
        for _idev in range(ndev):
            _start = _idev * _di
            _end = (_idev + 1) * _di
            gate_up.append(mlp_gate[_start:_end, :])
            gate_up.append(mlp_up[_start:_end, :])
        mlp_gate_up = torch.cat(gate_up, dim=0).to(meta.torch_dtype).contiguous()

        weight_loader.contents.lang_loader.load_mlp_gate_up(weights, mlp_gate_up.data_ptr(), i)
        del mlp_gate_up

        mlp_down = load_specific_tensor(model_path, lang_names.mlp_down(i))
        mlp_down = mlp_down.to(meta.torch_dtype).reshape([d, ndev, di // ndev]).transpose(0, 1).contiguous()
        weight_loader.contents.lang_loader.load_mlp_down(weights, mlp_down.data_ptr(), i)
        del mlp_down

    # -------------------------------
    # Vision head weights
    # -------------------------------
    patch_embed_weight = load_specific_tensor(model_path, vis_names.patch_embed_weight()).to(meta.torch_dtype)
    weight_loader.contents.vis_loader.load_patch_embed_weight(weights, patch_embed_weight.data_ptr())
    del patch_embed_weight

    patch_embed_bias = load_specific_tensor(model_path, vis_names.patch_embed_bias()).to(meta.torch_dtype)
    weight_loader.contents.vis_loader.load_patch_embed_bias(weights, patch_embed_bias.data_ptr())
    del patch_embed_bias

    pos_embed_weight = load_specific_tensor(model_path, vis_names.pos_embed_weight()).to(meta.torch_dtype)
    weight_loader.contents.vis_loader.load_pos_embed_weight(weights, pos_embed_weight.data_ptr())
    del pos_embed_weight

    for i in range(meta.vis_meta.depth):
        attn_proj_weight = load_specific_tensor(model_path, vis_names.attn_proj_weight(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_attn_proj_weight(weights, attn_proj_weight.data_ptr(), i)
        del attn_proj_weight

        attn_proj_bias = load_specific_tensor(model_path, vis_names.attn_proj_bias(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_attn_proj_bias(weights, attn_proj_bias.data_ptr(), i)
        del attn_proj_bias

        attn_qkv_weight = load_specific_tensor(model_path, vis_names.attn_qkv_weight(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_attn_qkv_weight(weights, attn_qkv_weight.data_ptr(), i)
        del attn_qkv_weight

        attn_qkv_bias = load_specific_tensor(model_path, vis_names.attn_qkv_bias(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_attn_qkv_bias(weights, attn_qkv_bias.data_ptr(), i)
        del attn_qkv_bias

        mlp_linear_fc1_weight = load_specific_tensor(model_path, vis_names.mlp_linear_fc1_weight(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_mlp_linear_fc1_weight(weights, mlp_linear_fc1_weight.data_ptr(), i)
        del mlp_linear_fc1_weight

        mlp_linear_fc1_bias = load_specific_tensor(model_path, vis_names.mlp_linear_fc1_bias(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_mlp_linear_fc1_bias(weights, mlp_linear_fc1_bias.data_ptr(), i)
        del mlp_linear_fc1_bias

        mlp_linear_fc2_weight = load_specific_tensor(model_path, vis_names.mlp_linear_fc2_weight(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_mlp_linear_fc2_weight(weights, mlp_linear_fc2_weight.data_ptr(), i)
        del mlp_linear_fc2_weight

        mlp_linear_fc2_bias = load_specific_tensor(model_path, vis_names.mlp_linear_fc2_bias(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_mlp_linear_fc2_bias(weights, mlp_linear_fc2_bias.data_ptr(), i)
        del mlp_linear_fc2_bias

        norm1_weight = load_specific_tensor(model_path, vis_names.norm1_weight(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_norm1_weight(weights, norm1_weight.data_ptr(), i)
        del norm1_weight

        norm1_bias = load_specific_tensor(model_path, vis_names.norm1_bias(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_norm1_bias(weights, norm1_bias.data_ptr(), i)
        del norm1_bias

        norm2_weight = load_specific_tensor(model_path, vis_names.norm2_weight(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_norm2_weight(weights, norm2_weight.data_ptr(), i)
        del norm2_weight

        norm2_bias = load_specific_tensor(model_path, vis_names.norm2_bias(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_norm2_bias(weights, norm2_bias.data_ptr(), i)
        del norm2_bias

    for i in range(len(meta.vis_meta.deepstack_visual_indexes)):
        deepstack_merger_linear_fc1_weight = load_specific_tensor(model_path, vis_names.deepstack_merger_linear_fc1_weight(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_deepstack_merger_linear_fc1_weight(weights, deepstack_merger_linear_fc1_weight.data_ptr(), i)
        del deepstack_merger_linear_fc1_weight

        deepstack_merger_linear_fc1_bias = load_specific_tensor(model_path, vis_names.deepstack_merger_linear_fc1_bias(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_deepstack_merger_linear_fc1_bias(weights, deepstack_merger_linear_fc1_bias.data_ptr(), i)
        del deepstack_merger_linear_fc1_bias

        deepstack_merger_linear_fc2_weight = load_specific_tensor(model_path, vis_names.deepstack_merger_linear_fc2_weight(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_deepstack_merger_linear_fc2_weight(weights, deepstack_merger_linear_fc2_weight.data_ptr(), i)
        del deepstack_merger_linear_fc2_weight

        deepstack_merger_linear_fc2_bias = load_specific_tensor(model_path, vis_names.deepstack_merger_linear_fc2_bias(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_deepstack_merger_linear_fc2_bias(weights, deepstack_merger_linear_fc2_bias.data_ptr(), i)
        del deepstack_merger_linear_fc2_bias

        deepstack_merger_norm_weight = load_specific_tensor(model_path, vis_names.deepstack_merger_norm_weight(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_deepstack_merger_norm_weight(weights, deepstack_merger_norm_weight.data_ptr(), i)
        del deepstack_merger_norm_weight

        deepstack_merger_norm_bias = load_specific_tensor(model_path, vis_names.deepstack_merger_norm_bias(i)).to(meta.torch_dtype)
        weight_loader.contents.vis_loader.load_deepstack_merger_norm_bias(weights, deepstack_merger_norm_bias.data_ptr(), i)
        del deepstack_merger_norm_bias
    
    merger_linear_fc1_weight = load_specific_tensor(model_path, vis_names.merger_linear_fc1_weight()).to(meta.torch_dtype)
    weight_loader.contents.vis_loader.load_merger_linear_fc1_weight(weights, merger_linear_fc1_weight.data_ptr())
    del merger_linear_fc1_weight

    merger_linear_fc1_bias = load_specific_tensor(model_path, vis_names.merger_linear_fc1_bias()).to(meta.torch_dtype)
    weight_loader.contents.vis_loader.load_merger_linear_fc1_bias(weights, merger_linear_fc1_bias.data_ptr())
    del merger_linear_fc1_bias

    merger_linear_fc2_weight = load_specific_tensor(model_path, vis_names.merger_linear_fc2_weight()).to(meta.torch_dtype)
    weight_loader.contents.vis_loader.load_merger_linear_fc2_weight(weights, merger_linear_fc2_weight.data_ptr())
    del merger_linear_fc2_weight

    merger_linear_fc2_bias = load_specific_tensor(model_path, vis_names.merger_linear_fc2_bias()).to(meta.torch_dtype)
    weight_loader.contents.vis_loader.load_merger_linear_fc2_bias(weights, merger_linear_fc2_bias.data_ptr())
    del merger_linear_fc2_bias

    merger_norm_weight = load_specific_tensor(model_path, vis_names.merger_norm_weight()).to(meta.torch_dtype)
    weight_loader.contents.vis_loader.load_merger_norm_weight(weights, merger_norm_weight.data_ptr())
    del merger_norm_weight

    merger_norm_bias = load_specific_tensor(model_path, vis_names.merger_norm_bias()).to(meta.torch_dtype)
    weight_loader.contents.vis_loader.load_merger_norm_bias(weights, merger_norm_bias.data_ptr())
    del merger_norm_bias


class Qwen3vlBatchedTask:
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
        self.kv_caches = (POINTER(Qwen3vlCacheCStruct) * self.nreq)(
            *self.kv_cache_ptrs
        )
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

        # initialize visual encoder inputs
        self.pixel_values = None
        self.total_patches = 0
        self.image_grid_thw = None
        self.num_images = 0
        self.pixel_values_videos = None
        self.total_patches_videos = 0
        self.video_grid_thw = None
        self.num_videos = 0
        self.patch_features = 0

        # Prepare visual encoder inputs
        all_pixel_values = [t.inputs['pixel_values'] for t in tasks if 'pixel_values' in t.inputs]
        all_image_grid_thw = [t.inputs['image_grid_thw'] for t in tasks if 'image_grid_thw' in t.inputs]
        all_pixel_values_videos = [t.inputs['pixel_values_videos'] for t in tasks if 'pixel_values_videos' in t.inputs]
        all_video_grid_thw = [t.inputs['video_grid_thw'] for t in tasks if 'video_grid_thw' in t.inputs]

        if all_pixel_values:
            concat_pixel_values = torch.cat(all_pixel_values, dim=0)  # (total_patches, features)
            self.total_patches = concat_pixel_values.shape[0]
            self.patch_features = concat_pixel_values.shape[1]
            self.flat_pixels = concat_pixel_values.flatten().to(torch.bfloat16).contiguous()
            self.pixel_values = self.flat_pixels.ctypes.data_as(c_void_p)

        if all_image_grid_thw:
            concat_grid_thw = torch.cat(all_image_grid_thw, dim=0)  # (total_images, 3)
            self.num_images = concat_grid_thw.shape[0]
            flat_grid = concat_grid_thw.flatten().to(torch.int32).contiguous()
            self.image_grid_thw = (c_uint * len(flat_grid))(*flat_grid.tolist())

        if all_pixel_values_videos:
            concat_pixel_values_videos = torch.cat(all_pixel_values_videos, dim=0)  # (total_patches_videos, features)
            self.total_patches_videos = concat_pixel_values_videos.shape[0]
            self.patch_features_videos = concat_pixel_values_videos.shape[1]
            print(self.patch_features_videos, flush=True)
            self.flat_pixels_videos = concat_pixel_values_videos.flatten().to(torch.bfloat16).contiguous()
            self.pixel_values_videos = self.flat_pixels_videos.ctypes.data_as(c_void_p)

        if all_video_grid_thw:
            concat_grid_thw_videos = torch.cat(all_video_grid_thw, dim=0)  # (total_videos, 3)
            self.num_videos = concat_grid_thw_videos.shape[0]
            flat_grid_videos = concat_grid_thw_videos.flatten().to(torch.int32).contiguous()
            self.video_grid_thw = (c_uint * len(flat_grid_videos))(*flat_grid_videos.tolist())


    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.pixel_values,
            self.total_patches,
            self.image_grid_thw,
            self.num_images,
            self.pixel_values_videos,
            self.total_patches_videos,
            self.video_grid_thw,
            self.num_videos,
            self.patch_features,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.kv_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )

# 需要处理 visual encoder的cache 和 image video输入
class Qwen3vlForCauslLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        eos_token_id = self.config["text_config"]["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )

        print(model_dir_path)

        if "qwen3_vl" == config["model_type"]:
            self.meta = Qwen3vlMeta(
                config, max_tokens=max_tokens
            )
            self.processor = transformers.AutoProcessor.from_pretrained(model_dir_path)
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_path)
        else:
            raise ValueError("Unsupported model architecture")

        print(f"Creating model on {ndev} devices...")
        load_start_time = time.time()
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])

        self.model_instance = Qwen3vlModel()
        weights = self.model_instance.create_weights(
            byref(self.meta),
            device,
            ndev,
            dev_ids,
            c_bool(True)
        )
        print("Loading weights...")
        # Load weights from host
        load_Qwen3vl_weights(self.meta, weights, model_dir_path, ndev)
        # Create model instance
        self.model_ptr = self.model_instance.create_model(
            byref(self.meta),
            weights,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def max_context_len(self):
        return self.meta.text_meta.max_tokens

    def create_kv_cache(self):
        return self.model_instance.create_cache(self.model_ptr)

    def drop_kv_cache(self, kv_cache):
        self.model_instance.drop_cache(self.model_ptr, kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = Qwen3vlBatchedTask(tasks)
        self.model_instance.infer_batch(
            self.model_ptr,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        inputs = self.processor.apply_chat_template(
            conversation = [{"role": "user","content": [{"type": "text", "text": input_content}]}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        infer_task = InferTask(
            0,
            inputs,
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            self.eos_token_id,
        )
        infer_task.bind_kvcache(KVCache(self))
        print(input_content, end="", flush=True)
        steps = 0
        total_time = 0
        output_content = ""

        print(inputs['input_ids'][0].tolist(), flush=True)

        for step_i in range(max_steps):
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            print(output_tokens)
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
        avg_time = total_time * 1000 / steps if steps > 0 else -1
        print(output_content, flush=True)
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    def destroy_model_instance(self):
        self.model_instance.destroy_model(self.model_ptr)
        print("Model destroyed")


def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python qwen3vl.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
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
            "Usage: python qwen3vl.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = Qwen3vlForCauslLM(model_path, device_type, ndev, max_tokens=1024)
    model.generate("山东最高的山是？", 200)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()