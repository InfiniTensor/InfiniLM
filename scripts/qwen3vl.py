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
import numpy as np
from PIL import Image

from libinfinicore_infer import (
    Qwen3VLModel,
    Qwen3VLMetaCStruct,
    DataType,
    DeviceType,
    KVCacheCStruct,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref

torch.set_default_device("cpu")


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):
    """基于 transformers 的 smart_resize 实现"""
    import math
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio too large: {max(height, width) / min(height, width)}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


def preprocess_image_qwen3vl(image_path: str):
    """
    完整的 Qwen3-VL 图像预处理流程
    基于 transformers 的实现：加载→resize→rescale→normalize→CHW→reshape→permute→flatten
    """
    # 配置参数 (从 Qwen3-VL config 中获取)
    patch_size = 16
    merge_size = 2  # spatial_merge_size
    temporal_patch_size = 2
    factor = patch_size * merge_size  # 28
    min_pixels = 4 * 28 * 28
    max_pixels = 16384 * 28 * 28

    # 1. 加载图像
    image = Image.open(image_path).convert('RGB')
    height, width = image.size[1], image.size[0]  # PIL: (width, height)

    # 2. Smart resize (保持宽高比，满足像素数和因子整除要求)
    resized_height, resized_width = smart_resize(height, width, factor, min_pixels, max_pixels)
    image = image.resize((resized_width, resized_height), Image.BILINEAR)

    print(f"图像预处理: {width}×{height} -> {resized_width}×{resized_height}")

    # 3. 转换为张量
    patches = torch.tensor(np.array(image)).float()

    # 4. Rescale (0-255 -> 0-1)
    patches = patches / 255.0

    # 5. Normalize (ImageNet 标准)
    # mean = torch.tensor([0.485, 0.456, 0.406])
    mean = torch.tensor([0.5, 0.5, 0.5])
    # std = torch.tensor([0.229, 0.224, 0.225])
    std = torch.tensor([0.5, 0.5, 0.5])
    patches = (patches - mean) / std

    # 6. CHW 调整: [H, W, C] -> [C, H, W]
    patches = patches.permute(2, 0, 1)

    # 7. 添加 batch 和时间维度 [C, H, W] -> [1, C, H, W] (模拟单帧)
    patches = patches.unsqueeze(0)

    # 8. Temporal padding (确保帧数能被 temporal_patch_size 整除)
    if patches.shape[0] % temporal_patch_size != 0:
        repeats = patches[-1:].repeat(temporal_patch_size - patches.shape[0] % temporal_patch_size, 1, 1, 1)
        patches = torch.cat([patches, repeats], dim=0)

    # 9. Grid 计算
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

    # 9. Reshape 和 Permute (按照 transformers 实现)
    patches = patches.view(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)

    # 10. Flatten
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
    )

    # Grid_thw (注意：这里是原始的 grid 大小，不是 merge 后的)
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int32)

    return flatten_patches, grid_thw


def compute_2d_mrope_pos_ids(grid_thw: torch.Tensor, spatial_merge_size: int = 2):
    """
    计算 2D MRoPE 的 pos_ids，基于 vLLM 的实现

    Args:
        grid_thw: [batch, 3] 张量，包含 [t, h, w] (原始 grid 大小)
        spatial_merge_size: 空间合并大小，默认2

    Returns:
        pos_ids: [num_patches, 2] 张量，包含 [h_pos, w_pos] 坐标
    """
    pos_ids_list = []

    for t, h, w in grid_thw:
        t, h, w = int(t), int(h), int(w)

        # 按照 vLLM 的 rot_pos_emb 实现，考虑 spatial_merge_size
        # 生成高度位置索引
        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()

        # 生成宽度位置索引
        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()

        # 组合坐标并重复时间维度
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        pos_ids_list.append(pos_ids)

    return torch.cat(pos_ids_list, dim=0)


class Qwen3VLMetaFromConfig(Qwen3VLMetaCStruct):
    def __init__(self, config, dtype=torch.bfloat16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_BF16

        self.scale_input = 1.0
        self.scale_output = 1.0
        self.scale_o = 1.0
        self.scale_down = 1.0
        has_qkv_bias = 0
        # 配置可能在顶层或在 text_config 中
        text_config = config.get("text_config", config)
        eos_token_id = text_config.get("eos_token_id")
        vision_config = config.get("vision_config", {})

        super().__init__(
            dt_logits=dt_,
            dt_linear_w=dt_,
            dt_norm_w=dt_,
            nlayer=text_config["num_hidden_layers"],
            d=text_config["hidden_size"],
            nh=text_config["num_attention_heads"],
            nkvh=text_config["num_key_value_heads"],
            dh=text_config["head_dim"],
            di=text_config["intermediate_size"],
            dctx=text_config["max_position_embeddings"] if max_tokens is None else max_tokens,
            dvoc=text_config["vocab_size"],
            epsilon=text_config["rms_norm_eps"],
            theta=text_config["rope_theta"],
            end_token=eos_token_id,
            has_qkv_bias=has_qkv_bias,
            # vision encoder parameters
            use_qk_norm=1 if text_config.get("use_qk_norm", False) else 0,
            vision_hidden_size=vision_config.get("hidden_size", 768),
            vision_layers=vision_config.get("depth", 12),
            vision_heads=vision_config.get("num_heads", 12),
            patch_size=vision_config.get("patch_size", 16),
            img_size=vision_config.get("img_size", 768),
            image_token_id=int(config.get("image_token_id", 151654)),
            video_token_id=int(config.get("video_token_id", 151656)),
        )
        self.torch_dtype_logits = dtype
        # 保留到python对象上，供上层使用
        try:
            self.image_token_id = int(config.get("image_token_id", 151654))
        except Exception:
            self.image_token_id = 151654
        try:
            self.video_token_id = int(config.get("video_token_id", 151656))
        except Exception:
            self.video_token_id = 151656


class Qwen3VLBatchedTask:
    def __init__(self, tasks: List[InferTask], image_path: str | None = None, video_path: str | None = None, config: dict | None = None):
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

        # Flatten token lists - 对于 ViT，tokens 实际上是 patch embeddings
        flat_tokens = [tok for toks in token_lists for tok in toks]

        # 统一：ntok 始终为文本 token 数；pixel_values 仅在 prefill(首轮，pos==0) 且有图像时提供
        self.ntok = len(flat_tokens)
        self.pixel_values = None
        self.num_patches = 0
        self.patch_dim = 0
        self.grid_thw = None
        self.image_path = image_path
        self.video_path = video_path
        # 从config中读取image/video token id（若存在）
        self.image_token_id = None
        self.video_token_id = None
        if isinstance(config, dict):
            self.image_token_id = config.get("image_token_id", None)
            self.video_token_id = config.get("video_token_id", None)
        # prefill 判断：仅当该 batch 中存在 pos==0 的请求且包含图像占位符时，计算像素与pos_ids
        any_prefill_with_image = False
        def is_image_tok(tid: int) -> bool:
            if self.image_token_id is not None:
                try:
                    return tid == int(self.image_token_id)
                except Exception:
                    pass
            return 151652 <= tid <= 151656
        def is_video_tok(tid: int) -> bool:
            if self.video_token_id is not None:
                try:
                    return tid == int(self.video_token_id)
                except Exception:
                    pass
            return tid == 151656

        for task in tasks:
            # print(f"[DEBUG] Task pos={task.pos}, tokens={task.tokens}")
            has_image_token = any(is_image_tok(token) or is_video_tok(token) for token in task.tokens)
            # print(f"[DEBUG] Has image/video token: {has_image_token}")
            if task.pos == 0 and has_image_token:
                any_prefill_with_image = True
                break
        # print(f"[DEBUG] any_prefill_with_image: {any_prefill_with_image}")
        # print(f"[DEBUG] image_path: {self.image_path}")
        if any_prefill_with_image:
            try:
                src_path = self.image_path if self.image_path is not None else self.video_path
                if src_path is None:
                    raise RuntimeError("no image/video path provided for prefill with vision input")
                # print(f"[DEBUG] Processing image: {src_path}")
                self.pixel_values, self.grid_thw = preprocess_image_qwen3vl(src_path)
                self.num_patches = self.pixel_values.shape[0]  # 设置 patch 数量
                self.patch_dim = self.pixel_values.shape[1]
                # print(f"[DEBUG] Pixel values shape: {self.pixel_values.shape}")
                # print(f"[DEBUG] Grid THW: {self.grid_thw}")
                # print(f"[DEBUG] Number of patches: {self.num_patches}")
            except Exception as _e:
                self.pixel_values = None
                self.grid_thw = None
                self.num_patches = 0
                self.patch_dim = 0

        # 实现 2D MRoPE pos_ids 计算
        # 集成图像 pos_ids 到批处理中
        flat_pos_ids = []
        self.has_vision = False  # 默认无视觉输入
        self.vision_pos_shape = None

        if any_prefill_with_image and getattr(self, 'pixel_values', None) is not None and getattr(self, 'grid_thw', None) is not None:
            try:
                pos_ids = compute_2d_mrope_pos_ids(self.grid_thw)
                for pos in pos_ids:
                    flat_pos_ids.extend([int(pos[0].item()), int(pos[1].item())])
                self.has_vision = True
                self.vision_pos_shape = pos_ids.shape
            except Exception as e:
                print(f"警告: 图像 pos_ids 计算失败，prefill 将降级为纯文本: {e}")
                self.has_vision = False
        else:
            # 纯文本或decode：为每个token提供简化pos_ids，避免C端空指针
            for toks in token_lists:
                for i in range(len(toks)):
                    flat_pos_ids.extend([i, 0])
            self.has_vision = False

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        # 确保 flat_pos_ids 都是整数并构造 ctypes 数组
        safe_pos_ids = [int(x) for x in flat_pos_ids] if flat_pos_ids else [0]
        self.pos_ids = (c_uint * len(safe_pos_ids))(*safe_pos_ids)
        self.pos_ids_len = len(safe_pos_ids)
        self.kv_caches = (POINTER(KVCacheCStruct) *
                          self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

        # 构造 3D mRoPE 参数
        self.llm_pos_ids = None
        self.llm_pos_ids_len = 0
        self.rope_section = None
        self.rope_section_len = 0

        # 构造 deepstack_layers 参数
        vision_config = getattr(config, 'vision_config', {})
        deepstack_visual_indexes = getattr(vision_config, 'deepstack_visual_multiscale_indexes', [3, 6, 9])
        self.deepstack_layers = (c_uint * len(deepstack_visual_indexes))(*deepstack_visual_indexes)
        self.deepstack_layers_len = len(deepstack_visual_indexes)

        if self.has_vision and hasattr(self, 'num_patches') and self.num_patches > 0:
            # 构造 3D MRoPE pos_ids，考虑token替换：ntok-1+num_patches
            # 基于Rust代码的逻辑：pre_text + vision + post_text

            # 计算image token的位置（假设只有一个image token）
            image_token_id = 151655  # <|image_pad|> token
            image_token_pos = -1
            for i, token in enumerate(self.tokens):
                if token == image_token_id:
                    image_token_pos = i
                    break

            if image_token_pos == -1:
                # 如果没找到image token，按原逻辑处理
                pre_text_len = 0
                post_text_len = self.ntok
            else:
                pre_text_len = image_token_pos  # image token之前的文本
                post_text_len = self.ntok - image_token_pos - 1  # image token之后的文本

            # print(f"[DEBUG] 3D pos_ids: pre_text_len={pre_text_len}, vision_len={self.num_patches}, post_text_len={post_text_len}")

            total_len = pre_text_len + self.num_patches + post_text_len
            llm_pos_ids_flat = []

            # 图像前文本：每个维度都是连续递增
            for i in range(pre_text_len):
                llm_pos_ids_flat.extend([i, i, i])

            # 视觉部分：参考Rust代码的3D位置计算
            img_start_pos = pre_text_len
            # 简化处理：假设t=1, h=20, w=30 (对应600个patches)
            t_len, h_len, w_len = 1, 20, 30
            for t in range(t_len):
                for h in range(h_len):
                    for w in range(w_len):
                        t_pos = img_start_pos + t
                        h_pos = img_start_pos + h
                        w_pos = img_start_pos + w
                        llm_pos_ids_flat.extend([t_pos, h_pos, w_pos])

            # 图像后文本：从视觉最大位置+1开始
            vision_max_pos = max(img_start_pos + t_len - 1,
                               img_start_pos + h_len - 1,
                               img_start_pos + w_len - 1)
            text_start_pos = vision_max_pos + 1
            for i in range(post_text_len):
                pos_val = text_start_pos + i
                llm_pos_ids_flat.extend([pos_val, pos_val, pos_val])

            self.llm_pos_ids_len = len(llm_pos_ids_flat)
            self.llm_pos_ids = (c_uint * self.llm_pos_ids_len)(*llm_pos_ids_flat)
            # print(f"[DEBUG] 构造的3D pos_ids长度: {self.llm_pos_ids_len//3}, 总长度: {self.llm_pos_ids_len}")

            # 构造 rope_section [3] = [t_max, h_max, w_max]
            # 从config读取，默认为[24, 20, 20]
            rope_section_vals = getattr(config, 'rope_scaling', {}).get('mrope_section', [24, 20, 20])
            self.rope_section_len = 3
            self.rope_section = (c_uint * 3)(*rope_section_vals)

    def input_args(self):
        # pixel_values 作为裸指针传递；无视觉输入则传空指针
        if getattr(self, 'pixel_values', None) is not None:
            # 确保是连续内存
            pv = self.pixel_values.contiguous()
            pixel_values_ptr = c_void_p(int(pv.data_ptr()))
        else:
            pixel_values_ptr = c_void_p(0)

        # 处理3D mRoPE参数的空指针
        llm_pos_ids_ptr = self.llm_pos_ids if self.llm_pos_ids is not None else POINTER(c_uint)()
        rope_section_ptr = self.rope_section if self.rope_section is not None else POINTER(c_uint)()

        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            self.pos_ids,
            self.pos_ids_len,
            llm_pos_ids_ptr,
            self.llm_pos_ids_len,
            rope_section_ptr,
            self.rope_section_len,
            pixel_values_ptr,
            self.kv_caches,
            self.temperaturas,
            self.topks,
            self.topps,
        )

    def get_vision_info(self):
        """获取视觉相关的信息，供 C++ 端使用"""
        return {
            'has_vision': getattr(self, 'has_vision', False),
            'vision_pos_shape': getattr(self, 'vision_pos_shape', None),
            'pos_ids_should_be_2d': getattr(self, 'has_vision', False)
        }


class Qwen3VLForCausalLM:
    def __init__(self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None):
        load_start_time = time.time()
        print(f"Creating model on {ndev} devices...")
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        # eos_token_id 可能在顶层或在 text_config 中
        eos_token_id = self.config.get("eos_token_id") or self.config.get("text_config", {}).get("eos_token_id")
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        self.dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.ndev = ndev
        self.device = device
        self.meta = Qwen3VLMetaFromConfig(config, max_tokens=max_tokens)

        self.qwen3vl_model = Qwen3VLModel()

        self.weights = self.qwen3vl_model.create_weights(
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

        self.model_instance = self.qwen3vl_model.create_model(
            byref(self.meta),
            self.weights,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def load_all_safetensors_from_dir(self, dir_path_: str):
        dir_path_ = Path(dir_path_)
        total_keys = 0

        # 创建检查文件夹
        check_dir = Path("./check")
        check_dir.mkdir(exist_ok=True)

        for file in sorted(dir_path_.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    total_keys += 1

                    tensor = f.get_tensor(key)

                    # 保存张量
                    self.save_tensor(key, tensor)

                    # if "o_proj.scales" in key:
                    #     tensor = tensor * self.meta.scale_o
                    # elif "down_proj.scales" in key:
                    #     tensor = tensor * self.meta.scale_down
                    # elif "embed_tokens.weight" in key:
                    #     tensor = tensor * self.meta.scale_input
                    # elif "lm_head.weight" in key:
                    #     tensor = tensor * self.meta.scale_output

                    self.qwen3vl_model.load_weight(
                        self.weights, key, tensor.data_ptr()
                    )
        print(f"加载的张量 key 总数: {total_keys}")

    def save_tensor(self, key: str, tensor, check_dir=None):
        if check_dir is None:
            check_dir = Path("./check")

        # 创建保存目录
        check_dir.mkdir(exist_ok=True)

        # 根据键名生成文件名
        filename = None

        # 1. Patch Embedding
        if key == "visual.patch_embed.proj.weight":
            filename = "1.patch_embd_w.txt"
        elif key == "visual.patch_embed.proj.bias":
            filename = "1.patch_embd_bias.txt"

        # 2. Position Embedding
        elif key == "visual.pos_embed.weight":
            filename = "2.pos_embd.txt"

        # 3. Block0 相关张量
        elif key == "visual.blocks.0.norm1.weight":
            filename = "3.block0.norm1_w.txt"
        elif key == "visual.blocks.0.norm1.bias":
            filename = "3.block0.norm1.bias.txt"
        elif key == "visual.blocks.0.attn.qkv.weight":
            filename = "3.block0.attn.qkv_w.txt"
        elif key == "visual.blocks.0.attn.qkv.bias":
            filename = "3.block0.attn.qkv.bias.txt"
        elif key == "visual.blocks.0.attn.proj.weight":
            filename = "3.block0.attn.proj_w.txt"
        elif key == "visual.blocks.0.attn.proj.bias":
            filename = "3.block0.attn.proj.bias.txt"
        elif key == "visual.blocks.0.norm2.weight":
            filename = "4.block0.norm2_w.txt"
        elif key == "visual.blocks.0.norm2.bias":
            filename = "4.block0.norm2.bias.txt"
        elif key == "visual.blocks.0.mlp.linear_fc1.weight":
            filename = "4.block0.mlp.fc1_w.txt"
        elif key == "visual.blocks.0.mlp.linear_fc1.bias":
            filename = "4.block0.mlp.fc1.bias.txt"
        elif key == "visual.blocks.0.mlp.linear_fc2.weight":
            filename = "4.block0.mlp.fc2_w.txt"
        elif key == "visual.blocks.0.mlp.linear_fc2.bias":
            filename = "4.block0.mlp.fc2.bias.txt"

        # 5. Merger
        elif key == "visual.merger.norm.weight":
            filename = "5.merger.norm_w.txt"
        elif key == "visual.merger.norm.bias":
            filename = "5.merger.norm.bias.txt"
        elif key == "visual.merger.linear_fc1.weight":
            filename = "5.merger.fc1_w.txt"
        elif key == "visual.merger.linear_fc1.bias":
            filename = "5.merger.fc1.bias.txt"
        elif key == "visual.merger.linear_fc2.weight":
            filename = "5.merger.fc2_w.txt"
        elif key == "visual.merger.linear_fc2.bias":
            filename = "5.merger.fc2.bias.txt"

        # 兼容原有的merger键名
        elif key == "visual.merger.ln_q.weight":
            filename = "5.merger.ln_q_w.txt"
        elif key == "visual.merger.ln_q.bias":
            filename = "5.merger.ln_q_bias.txt"
        elif key == "visual.merger.mlp.0.weight":
            filename = "5.merger.mlp0_w.txt"
        elif key == "visual.merger.mlp.0.bias":
            filename = "5.merger.mlp0_bias.txt"
        elif key == "visual.merger.mlp.2.weight":
            filename = "5.merger.mlp2_w.txt"
        elif key == "visual.merger.mlp.2.bias":
            filename = "5.merger.mlp2_bias.txt"

        # 6. Deepstack Merger List (动态匹配)
        elif "visual.deepstack_merger_list." in key:
            # 提取索引号
            parts = key.split(".")
            if len(parts) >= 3:
                try:
                    idx = int(parts[2])  # deepstack_merger_list.{idx}.xxx
                    suffix = ".".join(parts[3:])
                    prefix = f"6.deepstack{idx}"

                    if suffix == "norm.weight":
                        filename = f"{prefix}.norm_w.txt"
                    elif suffix == "norm.bias":
                        filename = f"{prefix}.norm.bias.txt"
                    elif suffix == "linear_fc1.weight":
                        filename = f"{prefix}.fc1_w.txt"
                    elif suffix == "linear_fc1.bias":
                        filename = f"{prefix}.fc1.bias.txt"
                    elif suffix == "linear_fc2.weight":
                        filename = f"{prefix}.fc2_w.txt"
                    elif suffix == "linear_fc2.bias":
                        filename = f"{prefix}.fc2.bias.txt"
                except ValueError:
                    pass

        # 如果没有匹配的文件名，直接返回
        if filename is None:
            return

        filepath = check_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            # 写入形状信息
            shape_str = f"Shape: {tuple(tensor.shape)}\n"
            f.write(shape_str)

            # 写入步长信息
            strides_str = f"Strides: {tuple(tensor.stride())}\n"
            f.write(strides_str)

            # 获取PyTorch风格的字符串表示（不进行类型转换）
            tensor_str = str(tensor.detach().cpu())

            # 写入张量数据
            f.write(tensor_str)

    def max_context_len(self):
        return self.meta.dctx

    def create_kv_cache(self):
        return self.qwen3vl_model.create_kv_cache(
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
        self.qwen3vl_model.drop_kv_cache(kv_cache)

    def batch_infer_one_round(self, tasks: List[InferTask], image_path: str | None = None, video_path: str | None = None):
        output = (c_uint * len(tasks))()
        batch_inputs = Qwen3VLBatchedTask(tasks, image_path=image_path, video_path=video_path, config=self.config)
        self.qwen3vl_model.infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0, image_path=None, video_path=None):
        # 如果有图片，需要在内容中添加图片占位符
        if image_path is not None:
            input_content = f"<|vision_start|><|image_pad|><|vision_end|>{input_content}"

        input_content = self.tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": input_content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
        # print(f"[DEBUG] Generated tokens: {tokens}")
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
            # prefill: step 0，传入image/video；decode：后续步不传
            output_tokens = self.batch_infer_one_round(
                [infer_task],
                image_path=image_path if step_i == 0 else None,
                video_path=video_path if step_i == 0 else None,
            )
            end_time = time.time()
            steps += 1
            # output_str = (
            #     self.tokenizer._tokenizer.id_to_token(output_tokens[0])
            #     .replace("▁", " ")
            #     .replace("<0x0A>", "\n")
            # )
            output_str = self.tokenizer.decode(output_tokens[0])
            output_content += output_str
            print(f"[DEBUG] Step {step_i}: token_id={output_tokens[0]}, token='{output_str}', eos_tokens={self.eos_token_id}")
            print(output_str, end="", flush=True)
            if output_tokens[0] in self.eos_token_id:
                print(f"[DEBUG] EOS token detected, breaking generation loop")
                break
            infer_task.next(output_tokens[0])

            if step_i > 0:
                total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1)
        print(f"Time per step: {avg_time:.3f}ms")

        infer_task._kv_cache.drop(self)
        return output_content, avg_time

    def perplexity(self, test_sequences: List[Sequence[int]], batch_size=10):
        tasks = [InferTask(i, [], self.max_context_len(), 1.0, 1, 1.0, self.eos_token_id) for i in range(batch_size)]
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

            batch_inputs = Qwen3VLBatchedTask(tasks[:batch_id], image_path=None, config=self.config)
            logits = torch.zeros(
                (batch_inputs.ntok, self.meta.dvoc), dtype=self.meta.torch_dtype_logits
            )
            # 评测路径：decode阶段不传像素；传递pos_ids以保持mrope输入稳定
            # 简化forward_batch调用：使用input_args()展开，但需要插入decode专用的空pixel_values
            args = list(batch_inputs.input_args())
            args[11] = c_void_p(0)  # 替换pixel_values为空指针（decode阶段不传像素）

            self.qwen3vl_model.forward_batch(
                self.model_instance,
                *args,
                logits.data_ptr(),
            )

            logits = logits.float()
            token_ids = torch.tensor(true_tokens, dtype=torch.int64)  # [ntok,]
            log_probs = torch.nn.functional.log_softmax(
                logits, dim=-1)  # (ntok, vocab)
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
        self.qwen3vl_model.destroy_model(self.model_instance)
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

    # 首先测试 pos_ids 计算
    # test_pos_ids_calculation()

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_tokens = 1024
    model = Qwen3VLForCausalLM(model_path, device_type, ndev, max_tokens=max_tokens)
    image_path = "/home/cearx/qy/model/Qwen3-VL-2B-Vit-86M-0828/image3.jpg"
    model.generate("描述这张图片", 500, image_path=image_path)
    model.destroy_model_instance()


def test_pos_ids_calculation():
    """测试 2D MRoPE pos_ids 计算"""
    print("=== 测试 2D MRoPE pos_ids 计算 ===")

    # 测试图像路径
    image_path = "/home/cearx/qy/model/Qwen3-VL-2B-Vit-86M-0828/image3.jpg"

    try:
        # 预处理图像
        pixel_values, grid_thw = preprocess_image_qwen3vl(image_path)
        print(f"图像预处理完成:")
        print(f"  pixel_values shape: {pixel_values.shape}")
        print(f"  grid_thw: {grid_thw}")

        # 计算 pos_ids
        pos_ids = compute_2d_mrope_pos_ids(grid_thw)
        print(f"pos_ids 计算完成:")
        print(f"  pos_ids shape: {pos_ids.shape}")
        print(f"  pos_ids 前10个元素:")
        print(f"  {pos_ids[:10]}")
        print(f"  pos_ids 最后10个元素:")
        print(f"  {pos_ids[-10:]}")

        # 验证 pos_ids 的合理性
        t, h, w = grid_thw[0].tolist()
        # grid_thw 现在已经是 spatial merge 后的网格大小
        expected_patches = t * h * w
        actual_patches = pos_ids.shape[0]
        print(f"期望 patch 数量: {expected_patches}")
        print(f"实际 patch 数量: {actual_patches}")

        # 检查 pos_ids 的值范围
        h_max = pos_ids[:, 0].max().item()
        w_max = pos_ids[:, 1].max().item()
        print(f"pos_ids 高度范围: 0 到 {h_max}")
        print(f"pos_ids 宽度范围: 0 到 {w_max}")

        # 坐标范围应该对应 grid_thw 的范围
        expected_h_max = h - 1
        expected_w_max = w - 1
        print(f"预期高度范围: 0 到 {expected_h_max}")
        print(f"预期宽度范围: 0 到 {expected_w_max}")

        if expected_patches == actual_patches:
            print("✓ pos_ids 数量验证通过!")
        else:
            print("✗ pos_ids 数量验证失败!")
            print(f"  详细信息: grid_thw={grid_thw}, 期望={expected_patches}, 实际={actual_patches}")

        # 检查坐标范围是否正确
        if h_max == expected_h_max and w_max == expected_w_max:
            print("✓ pos_ids 坐标范围验证通过!")
        else:
            print("✗ pos_ids 坐标范围验证失败!")
            print(f"  实际坐标最大值: h={h_max}, w={w_max}")
            print(f"  期望坐标最大值: h={expected_h_max}, w={expected_w_max}")

    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test()
