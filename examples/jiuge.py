import infinicore
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
import torch
from PIL import Image
from infinilm.cache import StaticKVCacheConfig, PagedKVCacheConfig, KVCompressionConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))


def get_args():
    parser = argparse.ArgumentParser(description="run Llama args")

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run cpu test",
    )
    parser.add_argument(
        "--nvidia",
        action="store_true",
        help="Run nvidia test",
    )
    parser.add_argument(
        "--metax",
        action="store_true",
        help="Run metax test",
    )
    parser.add_argument(
        "--moore",
        action="store_true",
        help="Run moore test",
    )
    parser.add_argument(
        "--iluvatar",
        action="store_true",
        help="Run iluvatar test",
    )
    parser.add_argument(
        "--cambricon",
        action="store_true",
        help="Run cambricon test",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="model_path",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="max_new_tokens",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpp",
        help="python or cpp model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="number of prompts in a batch",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="How are you",
        help="input prompt",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="image path for multimodal models",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="total rank for tensor parallel",
    )
    parser.add_argument(
        "--enable-paged-attn",
        action="store_true",
        help="use paged cache",
    )
    parser.add_argument(
        "--kv-compress",
        action="store_true",
        help="enable KV-cache compression (static cache only)",
    )
    parser.add_argument(
        "--kv-compress-weight",
        type=str,
        default="",
        help="path to KV compression .bin weights",
    )
    parser.add_argument(
        "--kv-compress-factor",
        type=int,
        default=1,
        help="compression factor (0/1 to use header)",
    )
    parser.add_argument(
        "--kv-compress-min-seq",
        type=int,
        default=0,
        help="minimum compressed sequence length (0 to use header)",
    )
    parser.add_argument(
        "--kv-image-kv-len",
        type=int,
        default=0,
        help="image KV prefix length (optional)",
    )
    parser.add_argument(
        "--no-stop-on-eos",
        action="store_true",
        help="disable early stop on eos for throughput measurement",
    )

    return parser.parse_args()


def test(
    prompts: str | list[str],
    model_path,
    max_new_tokens=100,
    infini_device=infinicore.device("cpu", 0),
    tp=1,
    enable_paged_attn=False,
    image_path=None,
    kv_compress_cfg=None,
    stop_on_eos=True,
):
    model_path = os.path.expanduser(model_path)
    # ---------------------------------------------------------------------------- #
    #                        Create Model
    # ---------------------------------------------------------------------------- #
    model = InferEngine(
        model_path,
        device=infini_device,
        distributed_config=DistConfig(tp),
    )

    # ---------------------------------------------------------------------------- #
    #                        Load Weights
    # ---------------------------------------------------------------------------- #
    load_model_state_dict_by_file(model, model_path, dtype=model.config.dtype)

    # ---------------------------------------------------------------------------- #
    #                        create tokenizer / processor
    # ---------------------------------------------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = None
    if image_path is not None:
        if model.config.model_type == "llava":
            from transformers import LlavaProcessor

            processor = LlavaProcessor.from_pretrained(model_path)
        elif model.config.model_type == "minicpmv":
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if "llama" == model.config.model_type:
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
            if model.config.model_type == "llava" and "<image>" not in prompt:
                prompt = "<image>\n" + prompt
            elif model.config.model_type == "minicpmv" and "<image>" not in prompt:
                prompt = "<image>./</image>\n" + prompt
            updated_prompts.append(prompt)
        prompts = updated_prompts
    input_contents = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]

    pixel_values = None
    image_bound = None
    tgt_sizes = None
    if image_path is not None and processor is not None:
        image = Image.open(image_path).convert("RGB")
        if model.config.model_type == "minicpmv":
            images = [[image] for _ in range(len(input_contents))]
        else:
            images = [image for _ in range(len(input_contents))]
        if model.config.model_type == "llava":
            inputs = processor(text=input_contents, images=images, return_tensors="pt")
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            input_ids_list = input_ids.tolist()
        elif model.config.model_type == "minicpmv":
            inputs = processor(
                text=input_contents,
                images=images,
                return_tensors="pt",
                max_slice_nums=1,
                use_image_id=False,
            )
            input_ids = inputs["input_ids"]
            input_ids_list = input_ids.tolist()
            pixel_values = inputs["pixel_values"]
            tgt_sizes = inputs["tgt_sizes"]
            image_bound = inputs["image_bound"]
        else:
            raise ValueError(f"Unsupported multimodal model_type: {model.config.model_type}")
    else:
        input_ids_list = tokenizer.batch_encode_plus(input_contents)[
            "input_ids"
        ]  # List: [[1, 1128, 526, 366, 29892]]

    # ---------------------------------------------------------------------------- #
    #                       Create KVCache
    # ---------------------------------------------------------------------------- #
    extra_cache_tokens = 0
    if image_path is not None and model.config.model_type == "llava":
        image_token_index = getattr(model.config, "image_token_index", None)
        if image_token_index is not None:
            num_image_tokens = int(input_ids_list[0].count(image_token_index))
            patch_size = model.config.vision_config.patch_size
            image_size = model.config.vision_config.image_size
            num_patches = (image_size // patch_size) ** 2
            extra_cache_tokens = num_image_tokens * (num_patches - 1)

    if enable_paged_attn:
        batch_size = 1 if prompts is str else len(prompts)
        max_total_tokens = max_new_tokens + len(input_ids_list[0])
        cache_config = PagedKVCacheConfig(
            num_blocks=(max_total_tokens // 16 + 1) * batch_size, block_size=16
        )
    else:
        batch_size = 1 if prompts is str else len(prompts)
        initial_capacity = max_new_tokens + len(input_ids_list[0]) + extra_cache_tokens
        cache_config = StaticKVCacheConfig(
            max_batch_size=batch_size, max_cache_len=initial_capacity
        )

    model.reset_cache(cache_config)

    # ---------------------------------------------------------------------------- #
    #                        Generate
    # ---------------------------------------------------------------------------- #
    print(input_contents[0], end="", flush=True)
    input_ids_infini = infinicore.from_list(input_ids_list)
    pixel_values_infini = None
    image_bound_infini = None
    tgt_sizes_infini = None

    if image_path is not None and processor is not None:
        if model.config.model_type == "llava":
            torch_dtype = infinicore.utils.to_torch_dtype(model.config.dtype)
            pixel_values_infini = infinicore.from_torch(
                pixel_values.to(dtype=torch_dtype)
            ).to(infini_device)
        elif model.config.model_type == "minicpmv":
            torch_dtype = infinicore.utils.to_torch_dtype(model.config.dtype)
            # Flatten pixel_values list-of-list; only support one slice per sample.
            if isinstance(pixel_values, list):
                pixel_values = [pv[0] if isinstance(pv, list) else pv for pv in pixel_values]
                pixel_values = torch.stack(pixel_values, dim=0)
            pixel_values_infini = infinicore.from_torch(
                pixel_values.to(dtype=torch_dtype)
            ).to(infini_device)

            # Pick image_bound ranges matching query_num.
            query_num = getattr(model.config, "query_num", 64)
            selected_bounds = []
            for bounds in image_bound:
                lengths = (bounds[:, 1] - bounds[:, 0]).tolist()
                keep = [i for i, l in enumerate(lengths) if l == query_num]
                if not keep:
                    keep = [int(np.argmax(lengths))]
                selected_bounds.append(bounds[keep])

            max_ranges = max(len(b) for b in selected_bounds)
            bound_np = np.zeros((len(selected_bounds), max_ranges, 2), dtype=np.int64)
            for i, bnd in enumerate(selected_bounds):
                if len(bnd) > 0:
                    bound_np[i, : len(bnd), :] = bnd.numpy()
            image_bound_infini = infinicore.from_numpy(bound_np)

            # tgt_sizes: use first entry per sample.
            tgt_np = np.stack([ts[0].numpy() for ts in tgt_sizes], axis=0).astype(np.int64)
            tgt_sizes_infini = infinicore.from_numpy(tgt_np)

    t1 = time.time()
    print("=================== start generate ====================")
    output_ids = model.generate(
        input_ids_infini,
        GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=1,
            top_k=1,
            top_p=0.8,
            stop_on_eos=stop_on_eos,
        ),
        _measure_and_log_time=True,
        pixel_values=pixel_values_infini,
        image_bound=image_bound_infini,
        tgt_sizes=tgt_sizes_infini,
        kv_compression_config=kv_compress_cfg,
    )
    t2 = time.time()

    numpy_output_ids = np.array([output_id.to_numpy()[0] for output_id in output_ids])
    print(tokenizer.decode(numpy_output_ids, skip_special_tokens=True))

    print(
        f"total_time: {round((t2 - t1) * 1000, 2)} ms",
    )


if __name__ == "__main__":
    args = get_args()
    print(args)

    # Parse command line arguments
    device_str = "cpu"
    if args.cpu:
        device_str = "cpu"
    elif args.nvidia:
        device_str = "cuda"
    elif args.metax:
        device_str = "cuda"
    elif args.moore:
        device_str = "musa"
    elif args.iluvatar:
        device_str = "cuda"
    elif args.cambricon:
        device_str = "mlu"
    else:
        print(
            "Usage:  python examples/jiuge.py [--cpu | --nvidia | --metax | --moore | --iluvatar] --model_path=<path/to/model_dir>\n"
            "such as, python examples/jiuge.py --nvidia --model_path=~/TinyLlama-1.1B-Chat-v1.0"
        )
        sys.exit(1)
    prompts = [args.prompt for _ in range(args.batch_size)]

    model_path = args.model_path
    max_new_tokens = args.max_new_tokens
    backend = args.backend
    tp = args.tp
    enable_paged_attn = args.enable_paged_attn
    kv_compress_cfg = None
    if args.kv_compress:
        kv_compress_cfg = KVCompressionConfig(
            enable=True,
            compression_factor=args.kv_compress_factor,
            min_seq_len=args.kv_compress_min_seq,
            image_kv_len=args.kv_image_kv_len,
            weight_path=args.kv_compress_weight,
        )
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
        image_path=args.image,
        kv_compress_cfg=kv_compress_cfg,
        stop_on_eos=not args.no_stop_on_eos,
    )
