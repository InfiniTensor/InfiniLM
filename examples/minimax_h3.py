#!/usr/bin/env python3

from __future__ import annotations

import argparse
import functools
import gc
import json
import re
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video
from infinilm.multimodal.minimax_h3 import (
    MiniMaxH3ConditionEncoder,
    MiniMaxH3DiffusersAdapter,
    MiniMaxH3TransformerRunner,
)
from safetensors.torch import load_file
from transformers import Qwen3VLForConditionalGeneration

DTYPE = torch.bfloat16
FPS = 24


class TimingReport:
    """Collect comparable wall-clock timings for both model backends."""

    DISPLAY_ORDER = (
        "model.pipeline_create",
        "model.condition_encoder_create_and_load",
        "model.transformer_create_and_load",
        "model.remaining_components_create_and_load",
        "input.files_and_references",
        "input.arguments",
        "compute.condition_encoder",
        "input.video_vae_encode",
        "input.audio_vae_encode",
        "compute.h3_transformer",
        "output.video_vae_decode",
        "output.audio_vae_decode",
        "generation.pipeline_total",
        "output.media_encode",
        "total.end_to_end",
    )

    def __init__(self, cuda_devices: Sequence[int]) -> None:
        self.cuda_devices = tuple(dict.fromkeys(cuda_devices))
        self.values: dict[str, list[float]] = defaultdict(list)

    def synchronize(self) -> None:
        if not torch.cuda.is_available():
            return
        for device in self.cuda_devices:
            torch.cuda.synchronize(device)

    @contextmanager
    def measure(self, name: str, *, synchronize: bool = False) -> Iterator[None]:
        if synchronize:
            self.synchronize()
        started = time.perf_counter()
        try:
            yield
        finally:
            if synchronize:
                self.synchronize()
            self.values[name].append(time.perf_counter() - started)

    def total(self, name: str) -> float:
        return sum(self.values.get(name, ()))

    def record(self, name: str, seconds: float) -> None:
        self.values[name].append(seconds)

    def print_report(self, pipeline_total: float) -> None:
        print("\n[timing] MiniMax-H3 profile")
        for name in self.DISPLAY_ORDER:
            samples = self.values.get(name, ())
            total = sum(samples)
            suffix = ""
            if len(samples) > 1:
                suffix = f" ({len(samples)} calls, {total / len(samples):.3f} s/call)"
            elif not samples:
                suffix = " (0 calls)"
            print(f"  {name:<38} {total:9.3f} s{suffix}")

        measured_compute = sum(
            self.total(name)
            for name in (
                "compute.condition_encoder",
                "compute.h3_transformer",
                "input.video_vae_encode",
                "input.audio_vae_encode",
                "output.video_vae_decode",
                "output.audio_vae_decode",
            )
        )
        residual = max(0.0, pipeline_total - measured_compute)
        print(f"  {'pipeline.other_preprocess_scheduler':<38} {residual:9.3f} s")


class ComponentProfiler:
    """Time the component methods invoked by the Diffusers pipeline."""

    def __init__(self, report: TimingReport) -> None:
        self.report = report
        self._originals: list[tuple[Any, str, Any]] = []

    def wrap(self, owner: Any, method_name: str, timing_name: str) -> None:
        original = getattr(owner, method_name)

        @functools.wraps(original)
        def timed(*args: Any, **kwargs: Any) -> Any:
            with self.report.measure(timing_name, synchronize=True):
                return original(*args, **kwargs)

        self._originals.append((owner, method_name, original))
        setattr(owner, method_name, timed)

    def close(self) -> None:
        while self._originals:
            owner, method_name, original = self._originals.pop()
            setattr(owner, method_name, original)


def install_component_profilers(
    report: TimingReport,
    *,
    text_encoder: Any,
    transformer: Any,
    vae: Any,
    audio_vae: Any,
) -> ComponentProfiler:
    profiler = ComponentProfiler(report)
    profiler.wrap(text_encoder.model, "forward", "compute.condition_encoder")
    profiler.wrap(transformer, "forward", "compute.h3_transformer")
    profiler.wrap(vae, "encode", "input.video_vae_encode")
    profiler.wrap(audio_vae, "encode", "input.audio_vae_encode")
    profiler.wrap(vae, "decode", "output.video_vae_decode")
    profiler.wrap(audio_vae, "decode", "output.audio_vae_decode")
    return profiler


def parse_gpu_list(value: str) -> list[int]:
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("GPU list must contain at least one device")
    if len(set(devices)) != len(devices):
        raise ValueError("GPU list must not contain duplicate devices")
    return devices


def make_max_memory(devices: Sequence[int], fraction: float) -> dict[int, str]:
    return {
        device: f"{int(torch.cuda.get_device_properties(device).total_memory / 1024**3 * fraction)}GiB"
        for device in devices
    }


def resolve_legacy_device_groups(
    devices: Sequence[int],
    text_devices_arg: str | None,
    transformer_devices_arg: str | None,
) -> tuple[list[int], list[int]]:
    if (text_devices_arg is None) != (transformer_devices_arg is None):
        raise ValueError(
            "--legacy-text-gpus and --legacy-transformer-gpus must be supplied together"
        )
    if text_devices_arg is not None:
        text_devices = parse_gpu_list(text_devices_arg)
        transformer_devices = parse_gpu_list(transformer_devices_arg)
    else:
        if len(devices) < 2:
            raise ValueError("legacy mode requires at least two GPUs")
        split = len(devices) // 2
        text_devices = list(devices[:split])
        transformer_devices = list(devices[split:])
    overlap = set(text_devices) & set(transformer_devices)
    if overlap:
        raise ValueError(
            f"legacy text and transformer GPU groups overlap: {sorted(overlap)}"
        )
    return text_devices, transformer_devices


def _legacy_h3_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_attention_heads": config["num_attention_heads"],
        "attention_head_dim": config["attention_head_dim"],
        "hidden_size": config["hidden_size"],
        "num_layers": config["num_layers"],
        "num_refiner_layers": config["token_refiner_num_layers"],
        "ffn_dim": config["ffn_hidden_size"],
        "in_channels": config["latents_dim"],
        "audio_in_channels": config["audio_latents_dim"],
        "patch_size": tuple(config["patch_size"]),
        "text_dim": config["text_dim"],
        "freq_dim": config["timestep_input_dim"],
        "time_embed_hidden_dim": config["time_embed_hidden_size"],
        "time_embed_dim": config["time_embed_dim"],
        "rope_freq_dim": config["rope_inv_freq_len"],
        "norm_eps": config["norm_eps"],
        "qk_norm_eps": config["qk_norm_eps"],
        "final_norm_eps": config["final_norm_eps"],
    }


def _remap_legacy_h3_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    if name == "rope.inv_freq":
        return {}
    name = re.sub(
        r"^token_refiner\.blocks\.(\d+)\.",
        r"token_refiner.refiner_blocks.\1.",
        name,
    )
    name = re.sub(r"^blocks\.(\d+)\.", r"transformer_blocks.\1.", name)
    for source, target in (
        ("video_patch_proj.", "proj_in."),
        ("audio_patch_proj.", "audio_proj_in."),
        ("condition_proj.", "context_embedder."),
        ("time_embedder.proj_in.", "time_embedder.linear_1."),
        ("time_embedder.proj_out.", "time_embedder.linear_2."),
        ("final_layer.norm.", "norm_out.norm."),
        ("final_layer.adaln_proj.linear.", "norm_out.linear."),
        ("final_layer.video_out.", "proj_out."),
        ("final_layer.audio_out.", "audio_proj_out."),
    ):
        if name.startswith(source):
            name = target + name[len(source) :]
            break
    name = name.replace(".attn.q_norm.", ".attn.norm_q.")
    name = name.replace(".attn.k_norm.", ".attn.norm_k.")
    name = name.replace(".attn.out_proj.", ".attn.to_out.0.")
    name = name.replace(".mlp.fc1.", ".ff.net.0.proj.")
    name = name.replace(".mlp.fc2.", ".ff.net.2.")

    if not name.endswith(".attn.qkv_proj.weight"):
        return {name: tensor}
    packed = tensor.view(num_heads, 3, head_dim, tensor.shape[1])
    query, key, value = packed.unbind(dim=1)
    prefix = name[: -len("qkv_proj.weight")]
    shape = (num_heads * head_dim, tensor.shape[1])
    return {
        prefix + "to_q.weight": query.reshape(shape),
        prefix + "to_k.weight": key.reshape(shape),
        prefix + "to_v.weight": value.reshape(shape),
    }


def _parameter_device(name: str, device_map: dict[str, Any]) -> Any:
    module_name = name
    while module_name not in device_map:
        if "." not in module_name:
            module_name = ""
            break
        module_name = module_name.rsplit(".", 1)[0]
    if module_name not in device_map:
        raise KeyError(f"No device-map entry covers {name}")
    return device_map[module_name]


def load_legacy_h3_transformer(
    transformer_dir: Path,
    devices: Sequence[int],
    memory_fraction: float,
) -> MiniMaxH3Transformer3DModel:
    """Stream an original H3 checkpoint into the current Diffusers layout."""

    with (transformer_dir / "config.json").open("r", encoding="utf-8") as handle:
        original_config = json.load(handle)
    model_config = _legacy_h3_config(original_config)
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel(**model_config)

    device_map = infer_auto_device_map(
        model,
        max_memory=make_max_memory(devices, memory_fraction),
        no_split_module_classes=model._no_split_modules,
        dtype=DTYPE,
        clean_result=True,
    )
    expected = set(model.state_dict())
    fp32_prefixes = tuple(model._keep_in_fp32_modules)
    with (transformer_dir / "model.safetensors.index.json").open(
        "r", encoding="utf-8"
    ) as handle:
        weight_map = json.load(handle)["weight_map"]

    for shard_name in dict.fromkeys(weight_map.values()):
        state_dict = load_file(transformer_dir / shard_name, device="cpu")
        for name, tensor in state_dict.items():
            remapped = _remap_legacy_h3_tensor(
                name,
                tensor,
                num_heads=model_config["num_attention_heads"],
                head_dim=model_config["attention_head_dim"],
            )
            for target_name, target_tensor in remapped.items():
                target_dtype = (
                    torch.float32 if target_name.startswith(fp32_prefixes) else DTYPE
                )
                set_module_tensor_to_device(
                    model,
                    target_name,
                    _parameter_device(target_name, device_map),
                    value=target_tensor,
                    dtype=target_dtype,
                )
                expected.remove(target_name)
        del state_dict
        gc.collect()

    if expected:
        missing = ", ".join(sorted(expected)[:10])
        raise RuntimeError(f"Legacy H3 checkpoint is missing parameters: {missing}")
    return dispatch_model(model, device_map=device_map, force_hooks=True)


@dataclass(frozen=True)
class ModelPaths:
    modular_dir: Path
    partition_dir: Path


def _read_model_index(path: Path) -> dict[str, Any]:
    index_path = path / "model_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing model_index.json under {path}")
    with index_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_model_paths(path: str, task: str) -> ModelPaths:
    candidate = Path(path).expanduser().resolve()
    if candidate.name == "diffusers":
        model_root = candidate.parent
        modular_dir = candidate
    elif candidate.name in ("FL2VA", "Ref2VA"):
        model_root = candidate.parent
        modular_dir = model_root / "diffusers"
    elif (candidate / "diffusers").is_dir():
        model_root = candidate
        modular_dir = candidate / "diffusers"
    else:
        model_root = candidate.parent
        modular_dir = candidate

    modular_index = _read_model_index(modular_dir)
    if modular_index.get("_class_name") != "MiniMaxH3ModularPipeline":
        raise ValueError(f"Expected a MiniMaxH3 modular pipeline under {modular_dir}")

    if task == "fl2va":
        partition_dir = modular_dir
    else:
        partition_dir = model_root / "Ref2VA"
        partition_index = _read_model_index(partition_dir)
        if partition_index.get("_minimax_h3", {}).get("partition") != "ref2va":
            raise ValueError(
                f"Expected a MiniMax-H3 Ref2VA checkpoint under {partition_dir}"
            )

    for component in ("text_encoder", "transformer"):
        if not (partition_dir / component).is_dir():
            raise FileNotFoundError(f"Missing {task} component: {component}")
    return ModelPaths(modular_dir=modular_dir, partition_dir=partition_dir)


def load_pipeline(
    paths: ModelPaths,
    task: str,
    devices: list[int],
    timings: TimingReport,
    *,
    legacy: bool,
    legacy_text_devices: list[int],
    legacy_transformer_devices: list[int],
    legacy_memory_fraction: float,
) -> tuple[ModularPipeline, torch.nn.Module, torch.nn.Module]:
    transformer_component = "transformer" if task == "fl2va" else "transformer_ref"
    workflow = "t2va" if task == "fl2va" else "ref2va"
    print(f"[model] Loading modular components from: {paths.modular_dir}")
    print(f"[model] Loading {task} weights from:       {paths.partition_dir}")
    print(f"[model] Backend: {'legacy PyTorch' if legacy else 'InfiniLM'}")
    if legacy:
        print(f"[model] Qwen3-VL devices: {legacy_text_devices}")
        print(f"[model] H3 transformer devices: {legacy_transformer_devices}")
    else:
        print(f"[model] InfiniLM TP devices: {devices}")

    with timings.measure("model.pipeline_create", synchronize=True):
        pipe = ModularPipeline.from_pretrained(
            str(paths.modular_dir), local_files_only=True
        )

    if legacy:
        with timings.measure(
            "model.condition_encoder_create_and_load", synchronize=True
        ):
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                str(paths.partition_dir),
                subfolder="text_encoder",
                dtype=DTYPE,
                device_map="balanced_low_0",
                max_memory=make_max_memory(legacy_text_devices, legacy_memory_fraction),
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        with timings.measure("model.transformer_create_and_load", synchronize=True):
            if task == "fl2va":
                transformer = MiniMaxH3Transformer3DModel.from_pretrained(
                    str(paths.partition_dir),
                    subfolder="transformer",
                    dtype=DTYPE,
                    device_map="balanced",
                    max_memory=make_max_memory(
                        legacy_transformer_devices, legacy_memory_fraction
                    ),
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            else:
                transformer = load_legacy_h3_transformer(
                    paths.partition_dir / "transformer",
                    legacy_transformer_devices,
                    legacy_memory_fraction,
                )
    else:
        with timings.measure(
            "model.condition_encoder_create_and_load", synchronize=True
        ):
            text_encoder = MiniMaxH3ConditionEncoder(
                paths.partition_dir / "text_encoder",
                dtype=DTYPE,
                tp_device_ids=devices,
            )
        with timings.measure("model.transformer_create_and_load", synchronize=True):
            runner = MiniMaxH3TransformerRunner(
                paths.partition_dir / "transformer",
                device="nvidia",
                tp_device_ids=devices,
            )
            transformer = MiniMaxH3DiffusersAdapter(runner, dtype=DTYPE)

    text_encoder.requires_grad_(False)
    text_encoder.eval()
    transformer.requires_grad_(False)
    transformer.eval()
    pipe.update_components(
        text_encoder=text_encoder,
        **{transformer_component: transformer},
    )

    with timings.measure(
        "model.remaining_components_create_and_load", synchronize=True
    ):
        pipe.load_components(
            workflow=workflow,
            dtype=DTYPE,
            pretrained_model_name_or_path=str(paths.modular_dir),
            local_files_only=True,
        )
        execution_device = pipe._execution_device
        print(f"[model] Diffusers execution device: {execution_device}")
        pipe.vae.to(device=execution_device, dtype=DTYPE)
        pipe.audio_vae.to(device=execution_device, dtype=DTYPE)
        pipe.vae.requires_grad_(False)
        pipe.audio_vae.requires_grad_(False)
    return pipe, text_encoder, transformer


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use the pure Transformers/Diffusers model implementation",
    )
    parser.add_argument(
        "--legacy-text-gpus",
        help="Legacy Qwen3-VL GPU list; defaults to the first half of --gpus",
    )
    parser.add_argument(
        "--legacy-transformer-gpus",
        help="Legacy H3 GPU list; defaults to the second half of --gpus",
    )
    parser.add_argument("--legacy-memory-fraction", type=float, default=0.82)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MiniMax-H3 FL2VA and Ref2VA generation"
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    fl2va = subparsers.add_parser(
        "fl2va", help="Text and optional first/last-frame generation"
    )
    add_common_arguments(fl2va)
    fl2va.add_argument("--first-frame")
    fl2va.add_argument("--last-frame")
    fl2va.add_argument("--num-frames", type=int, default=124)
    fl2va.add_argument("--output", default="minimax_h3_fl2va_output.mp4")

    ref2va = subparsers.add_parser(
        "ref2va", help="Reference image/video/audio conditioned generation"
    )
    add_common_arguments(ref2va)
    ref2va.add_argument("--reference-image")
    ref2va.add_argument("--reference-video")
    ref2va.add_argument("--audio")
    ref2va.add_argument("--num-frames", type=int)
    ref2va.add_argument("--output", default="minimax_h3_ref2va_output.mp4")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[int], list[int]]:
    devices = parse_gpu_list(args.gpus)
    legacy_text_devices: list[int] = []
    legacy_transformer_devices: list[int] = []
    if args.legacy:
        legacy_text_devices, legacy_transformer_devices = resolve_legacy_device_groups(
            devices,
            args.legacy_text_gpus,
            args.legacy_transformer_gpus,
        )
    active_devices = (
        legacy_text_devices + legacy_transformer_devices if args.legacy else devices
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if max(active_devices) >= torch.cuda.device_count():
        raise ValueError(
            f"Requested GPU {max(active_devices)}, but only "
            f"{torch.cuda.device_count()} are visible"
        )
    if not 0.1 <= args.legacy_memory_fraction <= 0.95:
        raise ValueError("--legacy-memory-fraction must be in [0.1, 0.95]")
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be supplied together")
    if args.height is not None and (args.height % 32 != 0 or args.width % 32 != 0):
        raise ValueError("--height and --width must be multiples of 32")
    if (
        args.task == "ref2va"
        and args.reference_image is None
        and args.reference_video is None
    ):
        raise ValueError("Ref2VA requires --reference-image or --reference-video")
    return devices, legacy_text_devices, legacy_transformer_devices


def prepare_generation_inputs(
    args: argparse.Namespace,
    timings: TimingReport,
) -> tuple[ModelPaths, dict[str, Any], dict[str, Any]]:
    with timings.measure("input.files_and_references"):
        paths = resolve_model_paths(args.model_path, args.task)
        if args.task == "fl2va":
            first_image = (
                load_image(args.first_frame).convert("RGB")
                if args.first_frame is not None
                else None
            )
            last_image = (
                load_image(args.last_frame).convert("RGB")
                if args.last_frame is not None
                else None
            )
            prepared = {
                "first_image": first_image,
                "last_image": last_image,
                "num_frames": args.num_frames,
            }
        else:
            references = []
            video_reference = None
            audio_reference = None
            if args.reference_image is not None:
                references.append(
                    MiniMaxH3ImageReference.from_file(args.reference_image)
                )
            if args.reference_video is not None:
                video_reference = MiniMaxH3VideoReference.from_file(
                    args.reference_video
                )
                references.append(video_reference)
            if args.audio is not None:
                audio_reference = MiniMaxH3AudioReference.from_file(args.audio)
                references.append(audio_reference)

            num_frames = args.num_frames
            if num_frames is None and audio_reference is not None:
                num_frames = round(
                    audio_reference.audio.shape[-1] / audio_reference.sample_rate * FPS
                )
            elif num_frames is None and video_reference is not None:
                num_frames = round(
                    len(video_reference.frames) / video_reference.fps * FPS
                )
            prepared = {"references": references, "num_frames": num_frames}

    with timings.measure("input.arguments"):
        generation_kwargs = {
            "prompt": args.text,
            "num_frames": prepared["num_frames"],
            "num_inference_steps": args.steps,
            "generator": torch.Generator("cpu").manual_seed(args.seed),
            "output": ["videos", "audio", "sampling_rate"],
        }
        if args.task == "fl2va":
            if prepared["first_image"] is not None:
                generation_kwargs["image"] = prepared["first_image"]
            if prepared["last_image"] is not None:
                generation_kwargs["last_image"] = prepared["last_image"]
        else:
            generation_kwargs["references"] = prepared["references"]
        if args.height is not None:
            generation_kwargs["height"] = args.height
            generation_kwargs["width"] = args.width
    return paths, generation_kwargs, prepared


def print_generation_summary(
    args: argparse.Namespace,
    prepared: dict[str, Any],
) -> None:
    print(f"[generate] Running MiniMax-H3 {args.task.upper()}")
    print(f"  text:            {args.text}")
    if args.task == "fl2va":
        print(f"  first frame:     {prepared['first_image'] is not None}")
        print(f"  last frame:      {prepared['last_image'] is not None}")
    else:
        if args.reference_image is not None:
            print(f"  reference image: {args.reference_image}")
        if args.reference_video is not None:
            print(f"  reference video: {args.reference_video}")
        if args.audio is not None:
            print(f"  reference audio: {args.audio}")
    print(f"  frames:          {prepared['num_frames']}")
    print(f"  steps:           {args.steps}")
    print(f"  backend:         {'legacy PyTorch' if args.legacy else 'InfiniLM'}")


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    devices, legacy_text_devices, legacy_transformer_devices = validate_args(args)
    active_devices = (
        legacy_text_devices + legacy_transformer_devices if args.legacy else devices
    )
    timings = TimingReport(active_devices)
    paths, generation_kwargs, prepared = prepare_generation_inputs(args, timings)
    pipe, text_encoder, transformer = load_pipeline(
        paths,
        args.task,
        devices,
        timings,
        legacy=args.legacy,
        legacy_text_devices=legacy_text_devices,
        legacy_transformer_devices=legacy_transformer_devices,
        legacy_memory_fraction=args.legacy_memory_fraction,
    )

    print_generation_summary(args, prepared)
    profiler = install_component_profilers(
        timings,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=pipe.vae,
        audio_vae=pipe.audio_vae,
    )
    try:
        with (
            torch.inference_mode(),
            timings.measure("generation.pipeline_total", synchronize=True),
        ):
            result = pipe(**generation_kwargs)
    finally:
        profiler.close()

    with timings.measure("output.media_encode"):
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        encode_video(
            result["videos"][0],
            fps=FPS,
            output_path=str(output_path),
            audio=result["audio"][0],
            audio_sample_rate=result["sampling_rate"],
        )
    timings.synchronize()
    timings.record("total.end_to_end", time.perf_counter() - total_started)
    timings.print_report(timings.total("generation.pipeline_total"))
    print(f"[done] Saved to: {output_path}")


if __name__ == "__main__":
    main()
