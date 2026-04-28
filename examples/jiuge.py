import infinicore
import transformers
from transformers import AutoTokenizer
from tokenizers import decoders as _dec
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
import argparse
import sys
import time
import os
import numpy as np
from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig
from packaging import version
from infinilm.base_config import BaseConfig

from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

_PAGED_KV_BLOCK_SIZE = 256


def test(
    prompts: str | list[str],
    model_path,
    max_new_tokens=100,
    infini_device=infinicore.device("cpu", 0),
    tp=1,
    enable_paged_attn=False,
    enable_graph=False,
    top_k=1,
    top_p=1.0,
    temperature=1.0,
    attn_backend="default",
    image_path=None,
):
    model_path = os.path.expanduser(model_path)
    # ---------------------------------------------------------------------------- #
    #                        Create Model
    # ---------------------------------------------------------------------------- #
    if enable_paged_attn and attn_backend == "default":
        attn_backend = "paged-attn"

    model = InferEngine(
        model_path,
        device=infini_device,
        distributed_config=DistConfig(tp),
        enable_graph_compiling=enable_graph,
        attention_backend=attn_backend,
        kv_cache_dtype=cfg.kv_cache_dtype,
    )
    # ---------------------------------------------------------------------------- #
    #                        Load Weights
    # ---------------------------------------------------------------------------- #
    load_model_state_dict_by_file(model, model_path, dtype=model.dtype)

    # ---------------------------------------------------------------------------- #
    #                        create tokenizer
    # ---------------------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    processor = None
    if image_path is not None:
        if model.model_type == "minicpmv":
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            tokenizer = processor.tokenizer

    if "llama" == model.model_type:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        target = getattr(backend, "_tokenizer", backend)
        norm = getattr(target, "normalizer", None)
        dec = getattr(target, "decoder", None)
        sn = repr(norm)[:800] if norm is not None else ""
        sd = repr(dec)[:800] if dec is not None else ""
        has_prepend = "Prepend" in sn
        has_strip = "Strip" in sd
        if has_prepend and has_strip:
            target.decoder = _dec.Sequence(
                [
                    _dec.Replace("▁", " "),
                    _dec.ByteFallback(),
                    _dec.Fuse(),
                ]
            )

    # ---------------------------------------------------------------------------- #
    #                        tokenize
    # ---------------------------------------------------------------------------- #
    # prompt = "山东最高的山是？"
    if isinstance(prompts, str):
        prompts = [prompts]

    if image_path is not None:
        updated_prompts = []
        for prompt in prompts:
            if model.model_type == "minicpmv" and "<image>" not in prompt:
                prompt = "(<image>./</image>)\n" + prompt
            updated_prompts.append(prompt)
        prompts = updated_prompts

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
        input_contents = [
            tokenizer.apply_chat_template(
                conversation=[{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]
    else:
        input_contents = prompts

    pixel_values = None
    image_bound = None
    tgt_sizes = None
    if image_path is not None and processor is not None:
        image = Image.open(image_path).convert("RGB")
        if model.model_type == "minicpmv":
            images = [[image] for _ in range(len(input_contents))]
        else:
            images = [image for _ in range(len(input_contents))]
        if model.model_type == "minicpmv":
            inputs = processor(
                text=input_contents,
                images=images,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"]
            input_ids_list = input_ids.tolist()
            pixel_values = inputs["pixel_values"]
            tgt_sizes = inputs["tgt_sizes"]
            image_bound = inputs["image_bound"]
        else:
            raise ValueError(f"Unsupported multimodal model_type: {model.model_type}")
    else:
        if hasattr(tokenizer, "batch_encode_plus"):
            input_ids_list = tokenizer.batch_encode_plus(input_contents)["input_ids"]
        else:
            input_ids_list = tokenizer(input_contents)["input_ids"]

        # input_ids_list = tokenizer.batch_encode_plus(input_contents)[
        #     "input_ids"
        # ]  # List: [[1, 1128, 526, 366, 29892]]
        if version.parse(transformers.__version__) < version.parse("5.0.0"):
            # Ideally this is solved by upgrading transformers. However, doing so causes version mismatch between transformers and mlu pytorch on devices with Phytium CPU. So a branch is temporarily used.
            input_ids_list = [
                tokenizer.encode_plus(
                    text, truncation=True, max_length=2048, add_special_tokens=True
                )["input_ids"]
                for text in input_contents
            ]
        else:
            input_ids_list = [
                tokenizer._encode_plus(
                    text, truncation=True, max_length=2048, add_special_tokens=True
                )["input_ids"]
                for text in input_contents
            ]

    # ---------------------------------------------------------------------------- #
    #                       Create KVCache
    # ---------------------------------------------------------------------------- #
    if enable_paged_attn:
        batch_size = 1 if prompts is str else len(prompts)
        max_total_tokens = max_new_tokens + len(input_ids_list[0])
        cache_config = PagedKVCacheConfig(
            num_blocks=(
                (max_total_tokens + (_PAGED_KV_BLOCK_SIZE - 1)) // _PAGED_KV_BLOCK_SIZE
            )
            * batch_size,
            block_size=_PAGED_KV_BLOCK_SIZE,
        )
    else:
        batch_size = 1 if prompts is str else len(prompts)
        initial_capacity = max_new_tokens + len(input_ids_list[0])
        cache_config = StaticKVCacheConfig(
            max_batch_size=batch_size, max_cache_len=initial_capacity
        )

    model.reset_cache(cache_config)

    # ---------------------------------------------------------------------------- #
    #                        Generate
    # ---------------------------------------------------------------------------- #
    print(input_contents[0], end="", flush=True)
    input_ids_infini = infinicore.from_list(input_ids_list)

    # Process multimodal inputs if needed
    pixel_values_infini = None
    image_bound_infini = None
    tgt_sizes_infini = None
    if image_path is not None and processor is not None:
        # TODO: Factor out this part per future multimodal model support.
        if model.model_type == "minicpmv":
            torch_dtype = infinicore.utils.to_torch_dtype(model.dtype)

            # 1. Pixel values
            all_pixel_values = []
            assert (
                len(pixel_values) == 1
            ), "Only batch_size=1 is supported yet for image inputs."
            for pv in pixel_values:
                all_pixel_values.extend(
                    [i.flatten(end_dim=1).permute(1, 0) for i in pv]
                )

            pixel_values_tensor = torch.nn.utils.rnn.pad_sequence(
                all_pixel_values, batch_first=True, padding_value=0.0
            ).to(dtype=torch_dtype)
            B, L, _ = pixel_values_tensor.shape
            pixel_values_tensor = (
                pixel_values_tensor.permute(0, 2, 1).reshape(B, 3, -1, L).contiguous()
            )
            pixel_values_infini = infinicore.from_torch(pixel_values_tensor)

            # 2. tgt_sizes
            all_tgt_sizes = [
                tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)
            ]

            tgt_sizes_tensor = torch.vstack(all_tgt_sizes).to(torch.int64)

            tgt_sizes_infini = infinicore.from_torch(tgt_sizes_tensor)

            # 3. image_bound
            batch_size = len(image_bound)
            max_ranges = max(len(b) for b in image_bound)

            bound_np = np.zeros((batch_size, max_ranges, 2), dtype=np.int64)

            for i, bnd in enumerate(image_bound):
                if len(bnd) > 0:
                    bound_np[i, : len(bnd), :] = bnd.cpu().numpy()

            image_bound_infini = infinicore.from_numpy(bound_np)

    t1 = time.time()
    print("=================== start generate ====================")
    output_ids = model.generate(
        input_ids_infini,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ),
        _measure_and_log_time=True,
        pixel_values=pixel_values_infini,
        image_bound=image_bound_infini,
        tgt_sizes=tgt_sizes_infini,
    )
    t2 = time.time()

    numpy_output_ids = np.array([output_id.to_numpy()[0] for output_id in output_ids])
    print(tokenizer.decode(numpy_output_ids, skip_special_tokens=True))

    print(
        f"total_time: {round((t2 - t1) * 1000, 2)} ms",
    )


if __name__ == "__main__":
    cfg = BaseConfig()

    device_str = cfg.get_device_str(cfg.device)

    prompts = [cfg.prompt for _ in range(cfg.batch_size)]

    _PAGED_KV_BLOCK_SIZE = cfg.block_size

    model_path = cfg.model

    max_new_tokens = cfg.max_new_tokens

    backend = cfg.backend

    tp = cfg.tp

    enable_paged_attn = cfg.enable_paged_attn

    enable_graph = cfg.enable_graph

    if backend != "cpp":
        raise ValueError(f"Unsupported backend: {backend}.")

    infini_device = infinicore.device(device_str, 0)

    test(
        prompts,
        model_path,
        max_new_tokens,
        infini_device=infini_device,
        tp=tp,
        enable_paged_attn=enable_paged_attn,
        enable_graph=enable_graph,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        temperature=cfg.temperature,
        attn_backend=cfg.attn,
        image_path=cfg.image,
    )
