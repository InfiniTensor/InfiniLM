#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

FL2VA_PROMPT = "A dog falls asleep"
REF2VA_PROMPT = (
    "Transform the reference video into a colorful hand-drawn cartoon animation style"
)
SEED = 42
STEPS = 50
NUM_FRAMES = 124

TIMING_ORDER = (
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
    "pipeline.other_preprocess_scheduler",
    "process.wall",
)
TIMING_PATTERN = re.compile(r"^\s{2}([a-z0-9_.]+)\s+([0-9]+(?:\.[0-9]+)?)\s+s(?:\s|$)")


@dataclass
class TaskResult:
    name: str
    backend: str
    status: str
    command: list[str]
    output_path: str
    log_path: str | None
    returncode: int | None
    wall_seconds: float
    timings: dict[str, float]
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.status in {"success", "provided"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy PyTorch and InfiniLM MiniMax-H3 generation"
    )
    parser.add_argument(
        "--model-path",
        help="Location of model",
    )
    parser.add_argument(
        "--first-frame",
        help="First frame required when an FL2VA output must be generated",
    )
    parser.add_argument(
        "--reference-video",
        help="Shared Ref2VA input required when running only the Ref2VA task",
    )
    parser.add_argument(
        "--output-dir",
        default="minimax_h3_comparison",
    )
    parser.add_argument(
        "--legacy-gpus",
        default="0,1,2,3",
        help="GPUs used by the PyTorch reference backend (default: 0,1,2,3)",
    )
    parser.add_argument(
        "--legacy-text-gpus",
        help="Optional explicit Qwen3-VL GPU list for the reference backend",
    )
    parser.add_argument(
        "--legacy-transformer-gpus",
        help="Optional explicit H3 transformer GPU list for the reference backend",
    )
    parser.add_argument(
        "--legacy-memory-fraction",
        type=float,
        default=0.4,
        help="Per-device placement budget for the reference backend (default: 0.4)",
    )
    parser.add_argument("--infinilm-gpus", default="0,1,2,3")
    parser.add_argument(
        "--tasks",
        default="fl2va,ref2va",
        help="Comma-separated tasks to run or compare: fl2va, ref2va",
    )
    parser.add_argument(
        "--reference-fl2va-video",
        help="Existing PyTorch FL2VA result; skips its generation",
    )
    parser.add_argument(
        "--infinilm-fl2va-video",
        help="Existing InfiniLM FL2VA result; skips its generation",
    )
    parser.add_argument(
        "--reference-ref2va-video",
        help="Existing PyTorch Ref2VA result; skips its generation",
    )
    parser.add_argument(
        "--infinilm-ref2va-video",
        help="Existing InfiniLM Ref2VA result; skips its generation",
    )
    return parser.parse_args()


def parse_tasks(value: str) -> tuple[str, ...]:
    tasks = tuple(
        dict.fromkeys(part.strip().lower() for part in value.split(",") if part.strip())
    )
    invalid = sorted(set(tasks) - {"fl2va", "ref2va"})
    if invalid:
        raise ValueError(f"unsupported tasks: {', '.join(invalid)}")
    if not tasks:
        raise ValueError("--tasks must select fl2va, ref2va, or both")
    return tasks


def resolve_legacy_launch(args: argparse.Namespace) -> tuple[str, float]:
    if (args.legacy_text_gpus is None) != (args.legacy_transformer_gpus is None):
        raise ValueError(
            "--legacy-text-gpus and --legacy-transformer-gpus must be supplied together"
        )

    legacy_gpus = args.legacy_gpus
    if not legacy_gpus:
        raise ValueError("--legacy-gpus must select at least one GPU")

    memory_fraction = args.legacy_memory_fraction
    if not 0.1 <= memory_fraction <= 0.95:
        raise ValueError("--legacy-memory-fraction must be in [0.1, 0.95]")

    print(
        f"[config] PyTorch reference gpus={legacy_gpus}, "
        f"memory_fraction={memory_fraction}"
    )
    return legacy_gpus, memory_fraction


def parse_timing_line(line: str, timings: dict[str, float]) -> None:
    match = TIMING_PATTERN.match(line)
    if match is not None:
        timings[match.group(1)] = float(match.group(2))


def run_task(
    *,
    name: str,
    backend: str,
    command: list[str],
    output_path: Path,
    log_path: Path,
    cwd: Path,
) -> TaskResult:
    output_path.unlink(missing_ok=True)
    timings: dict[str, float] = {}
    tail: deque[str] = deque(maxlen=40)
    started = time.perf_counter()

    print(f"\n[task] {name}")
    print(f"[task] command: {shlex.join(command)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
                tail.append(line.rstrip())
                parse_timing_line(line, timings)
            returncode = process.wait()
    except OSError as error:
        wall_seconds = time.perf_counter() - started
        message = f"failed to start child process: {error}"
        print(f"[task] ERROR: {message}")
        return TaskResult(
            name=name,
            backend=backend,
            status="failed",
            command=command,
            output_path=str(output_path),
            log_path=str(log_path),
            returncode=None,
            wall_seconds=wall_seconds,
            timings={"process.wall": wall_seconds},
            error=message,
        )

    wall_seconds = time.perf_counter() - started
    timings["process.wall"] = wall_seconds
    error = None
    status = "success"
    if returncode != 0:
        status = "failed"
        error = f"child exited with code {returncode}"
    elif not output_path.is_file():
        status = "failed"
        error = "child exited successfully but did not create its output video"

    if error is not None:
        tail_text = "\n".join(tail)
        if tail_text:
            error = f"{error}\nLast child output:\n{tail_text}"
        print(f"[task] ERROR: {error}")
    else:
        print(f"[task] completed in {wall_seconds:.3f} s: {output_path}")

    return TaskResult(
        name=name,
        backend=backend,
        status=status,
        command=command,
        output_path=str(output_path),
        log_path=str(log_path),
        returncode=returncode,
        wall_seconds=wall_seconds,
        timings=timings,
        error=error,
    )


def skipped_task(
    *,
    name: str,
    backend: str,
    output_path: Path,
    log_path: Path,
    reason: str,
) -> TaskResult:
    print(f"\n[task] {name}: SKIPPED: {reason}")
    return TaskResult(
        name=name,
        backend=backend,
        status="skipped",
        command=[],
        output_path=str(output_path),
        log_path=str(log_path),
        returncode=None,
        wall_seconds=0.0,
        timings={},
        error=reason,
    )


def provided_task(*, name: str, backend: str, output_path: Path) -> TaskResult:
    if not output_path.is_file():
        reason = f"provided result does not exist: {output_path}"
        print(f"\n[task] {name}: ERROR: {reason}")
        status = "failed"
    else:
        reason = None
        status = "provided"
        print(f"\n[task] {name}: using existing result: {output_path}")
    return TaskResult(
        name=name,
        backend=backend,
        status=status,
        command=[],
        output_path=str(output_path),
        log_path=None,
        returncode=None,
        wall_seconds=0.0,
        timings={},
        error=reason,
    )


def decoded_video_frames(path: Path) -> Iterator[Any]:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"no video stream found in {path}")
        yield from container.decode(video=0)


def probe_video(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"no video stream found in {path}")
        stream = container.streams.video[0]
        duration_seconds = None
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration / av.time_base)
        return {
            "width": stream.width,
            "height": stream.height,
            "declared_frames": stream.frames or None,
            "average_fps": (
                float(stream.average_rate) if stream.average_rate is not None else None
            ),
            "duration_seconds": duration_seconds,
        }


def prepare_ssim_frame(frame: Any, max_side: int = 512) -> Any:
    import numpy as np
    from PIL import Image

    image = Image.fromarray(frame).convert("L")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
    return np.asarray(image)


def compare_video_streams(left: Path, right: Path) -> dict[str, Any]:
    import numpy as np
    from skimage.metrics import structural_similarity

    left_metadata = probe_video(left)
    right_metadata = probe_video(right)
    sentinel = object()
    exact = True
    left_frames = 0
    right_frames = 0
    shape_mismatches = 0
    different_elements = 0
    compared_elements = 0
    absolute_difference_sum = 0
    max_absolute_difference = 0
    ssim_scores: list[float] = []

    for left_frame, right_frame in zip_longest(
        decoded_video_frames(left),
        decoded_video_frames(right),
        fillvalue=sentinel,
    ):
        if left_frame is not sentinel:
            left_frames += 1
        if right_frame is not sentinel:
            right_frames += 1
        if left_frame is sentinel or right_frame is sentinel:
            exact = False
            ssim_scores.append(0.0)
            continue

        left_array = left_frame.to_ndarray(format="rgb24")
        right_array = right_frame.to_ndarray(format="rgb24")
        if left_array.shape != right_array.shape:
            exact = False
            shape_mismatches += 1
            ssim_scores.append(0.0)
            continue

        ssim_scores.append(
            float(
                structural_similarity(
                    prepare_ssim_frame(left_array),
                    prepare_ssim_frame(right_array),
                    data_range=255,
                )
            )
        )
        difference = np.abs(left_array.astype(np.int16) - right_array.astype(np.int16))
        frame_differences = int(np.count_nonzero(difference))
        exact = exact and frame_differences == 0
        different_elements += frame_differences
        compared_elements += difference.size
        absolute_difference_sum += int(difference.sum(dtype=np.int64))
        max_absolute_difference = max(
            max_absolute_difference,
            int(difference.max()) if difference.size else 0,
        )

    return {
        "exact": exact,
        "left_metadata": left_metadata,
        "right_metadata": right_metadata,
        "metadata_size_match": (
            left_metadata["width"] == right_metadata["width"]
            and left_metadata["height"] == right_metadata["height"]
        ),
        "decoded_length_match": left_frames == right_frames,
        "left_frames": left_frames,
        "right_frames": right_frames,
        "shape_mismatches": shape_mismatches,
        "different_elements": different_elements,
        "compared_elements": compared_elements,
        "max_absolute_difference": max_absolute_difference,
        "mean_absolute_difference": (
            absolute_difference_sum / compared_elements if compared_elements else None
        ),
        "ssim_match_percent": (
            100.0 * sum(ssim_scores) / len(ssim_scores) if ssim_scores else None
        ),
        "ssim_min_percent": (100.0 * min(ssim_scores) if ssim_scores else None),
        "ssim_max_percent": (100.0 * max(ssim_scores) if ssim_scores else None),
        "ssim_method": "grayscale frames scaled to at most 512 pixels per side",
    }


def compare_task_outputs(
    label: str,
    reference: TaskResult,
    infinilm: TaskResult,
) -> dict[str, Any]:
    if not reference.available or not infinilm.available:
        reason = (
            f"reference status={reference.status}, InfiniLM status={infinilm.status}"
        )
        print(f"\n[compare] {label}: SKIPPED: {reason}")
        return {"status": "skipped", "reason": reason}

    print(f"\n[compare] {label}: decoding outputs")
    try:
        video = compare_video_streams(
            Path(reference.output_path), Path(infinilm.output_path)
        )
        result = {"status": "complete", "video": video}
    except Exception as error:
        result = {"status": "failed", "error": str(error)}
        print(f"[compare] ERROR: {result['error']}")
    return result


def print_timing_comparison(
    label: str,
    reference: TaskResult,
    infinilm: TaskResult,
) -> None:
    if not reference.timings and not infinilm.timings:
        return
    print(f"\n[timing comparison] {label}")
    print(
        f"  {'component':<42} {'PyTorch (s)':>12} {'InfiniLM (s)':>12} {'speedup':>10}"
    )
    print(f"  {'-' * 42} {'-' * 12} {'-' * 12} {'-' * 10}")
    extras = sorted(
        (set(reference.timings) | set(infinilm.timings)) - set(TIMING_ORDER)
    )
    for name in (*TIMING_ORDER, *extras):
        reference_time = reference.timings.get(name)
        infinilm_time = infinilm.timings.get(name)
        if reference_time is None and infinilm_time is None:
            continue
        reference_text = (
            f"{reference_time:.3f}" if reference_time is not None else "n/a"
        )
        infinilm_text = f"{infinilm_time:.3f}" if infinilm_time is not None else "n/a"
        speedup = (
            f"{reference_time / infinilm_time:.3f}x"
            if reference_time is not None
            and infinilm_time is not None
            and infinilm_time > 0
            else "n/a"
        )
        print(f"  {name:<42} {reference_text:>12} {infinilm_text:>12} {speedup:>10}")


def print_final_report(comparisons: dict[str, dict[str, Any]]) -> None:
    print("\n[result] MiniMax-H3 video comparison")
    for task, result in comparisons.items():
        label = task.upper()
        if result["status"] != "complete":
            detail = result.get("reason") or result.get("error", "unknown error")
            print(f"\n  {label}: {result['status'].upper()} ({detail})")
            continue

        video = result["video"]
        left = video["left_metadata"]
        right = video["right_metadata"]
        size_status = "PASS" if video["metadata_size_match"] else "FAIL"
        length_status = "PASS" if video["decoded_length_match"] else "FAIL"
        similarity = video["ssim_match_percent"]
        similarity_text = f"{similarity:.3f}%" if similarity is not None else "n/a"
        print(f"\n  {label}")
        print(
            f"    Size:         {size_status} "
            f"(PyTorch {left['width']}x{left['height']}, "
            f"InfiniLM {right['width']}x{right['height']})"
        )
        print(
            f"    Length:       {length_status} "
            f"(PyTorch {video['left_frames']} frames, "
            f"InfiniLM {video['right_frames']} frames)"
        )
        print(f"    Average SSIM: {similarity_text}")


def build_command(
    *,
    generator_script: Path,
    task: str,
    model_path: Path,
    gpus: str,
    output_path: Path,
    first_frame: Path | None = None,
    reference_video: Path | None = None,
    legacy: bool,
    legacy_text_gpus: str | None = None,
    legacy_transformer_gpus: str | None = None,
    legacy_memory_fraction: float | None = None,
) -> list[str]:
    prompt = FL2VA_PROMPT if task == "fl2va" else REF2VA_PROMPT
    command = [
        sys.executable,
        str(generator_script),
        task,
        f"--model-path={model_path}",
        f"--text={prompt}",
        f"--gpus={gpus}",
        f"--steps={STEPS}",
        f"--seed={SEED}",
        f"--num-frames={NUM_FRAMES}",
        f"--output={output_path}",
    ]
    if first_frame is not None:
        command.append(f"--first-frame={first_frame}")
    if reference_video is not None:
        command.append(f"--reference-video={reference_video}")
    if legacy:
        command.append("--legacy")
        if legacy_text_gpus is not None:
            command.append(f"--legacy-text-gpus={legacy_text_gpus}")
        if legacy_transformer_gpus is not None:
            command.append(f"--legacy-transformer-gpus={legacy_transformer_gpus}")
        if legacy_memory_fraction is not None:
            command.append(f"--legacy-memory-fraction={legacy_memory_fraction}")
    return command


def main() -> int:
    args = parse_args()
    try:
        selected_tasks = parse_tasks(args.tasks)
    except ValueError as error:
        print(f"[error] {error}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    generator_script = Path(__file__).with_name("minimax_h3.py").resolve()
    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_outputs = {
        "reference_fl2va": output_dir / "reference_fl2va.mp4",
        "reference_ref2va": output_dir / "reference_ref2va.mp4",
        "infinilm_fl2va": output_dir / "infinilm_fl2va.mp4",
        "infinilm_ref2va": output_dir / "infinilm_ref2va.mp4",
    }
    logs = {name: output_dir / f"{name}.log" for name in generated_outputs}
    provided_values = {
        "reference_fl2va": args.reference_fl2va_video,
        "reference_ref2va": args.reference_ref2va_video,
        "infinilm_fl2va": args.infinilm_fl2va_video,
        "infinilm_ref2va": args.infinilm_ref2va_video,
    }
    provided_outputs = {
        name: Path(value).expanduser().resolve() if value else None
        for name, value in provided_values.items()
    }
    selected_names = {
        f"{backend}_{task}"
        for task in selected_tasks
        for backend in ("reference", "infinilm")
    }
    needs_generation = any(provided_outputs[name] is None for name in selected_names)
    needs_fl2va_generation = "fl2va" in selected_tasks and any(
        provided_outputs[f"{backend}_fl2va"] is None
        for backend in ("reference", "infinilm")
    )
    needs_ref2va_generation = "ref2va" in selected_tasks and any(
        provided_outputs[f"{backend}_ref2va"] is None
        for backend in ("reference", "infinilm")
    )
    needs_legacy_generation = any(
        provided_outputs[f"reference_{task}"] is None for task in selected_tasks
    )

    legacy_gpus = args.legacy_gpus
    legacy_memory_fraction = args.legacy_memory_fraction
    if needs_legacy_generation:
        try:
            legacy_gpus, legacy_memory_fraction = resolve_legacy_launch(args)
        except ValueError as error:
            print(f"[error] {error}", file=sys.stderr)
            return 2

    if needs_generation:
        if not generator_script.is_file():
            print(f"[error] generator script does not exist: {generator_script}")
            return 2
        if not model_path.exists():
            print(f"[error] model path does not exist: {model_path}")
            return 2

    first_frame = (
        Path(args.first_frame).expanduser().resolve() if args.first_frame else None
    )
    if needs_fl2va_generation and (first_frame is None or not first_frame.is_file()):
        print("[error] --first-frame must name an existing file when generating FL2VA")
        return 2

    explicit_reference_video = (
        Path(args.reference_video).expanduser().resolve()
        if args.reference_video
        else None
    )
    if (
        selected_tasks == ("ref2va",)
        and needs_ref2va_generation
        and (explicit_reference_video is None or not explicit_reference_video.is_file())
    ):
        print(
            "[error] --reference-video must name an existing file when generating "
            "Ref2VA without FL2VA"
        )
        return 2

    task_results: dict[str, TaskResult] = {}

    if "fl2va" in selected_tasks:
        reference_output = provided_outputs["reference_fl2va"]
        if reference_output is not None:
            reference_fl2va = provided_task(
                name="reference_fl2va",
                backend="legacy_pytorch",
                output_path=reference_output,
            )
        else:
            assert first_frame is not None
            reference_fl2va = run_task(
                name="reference_fl2va",
                backend="legacy_pytorch",
                command=build_command(
                    generator_script=generator_script,
                    task="fl2va",
                    model_path=model_path,
                    gpus=legacy_gpus,
                    output_path=generated_outputs["reference_fl2va"],
                    first_frame=first_frame,
                    legacy=True,
                    legacy_text_gpus=args.legacy_text_gpus,
                    legacy_transformer_gpus=args.legacy_transformer_gpus,
                    legacy_memory_fraction=legacy_memory_fraction,
                ),
                output_path=generated_outputs["reference_fl2va"],
                log_path=logs["reference_fl2va"],
                cwd=repo_root,
            )
        task_results["reference_fl2va"] = reference_fl2va

    if "ref2va" in selected_tasks:
        shared_reference_video = (
            Path(task_results["reference_fl2va"].output_path)
            if "fl2va" in selected_tasks and task_results["reference_fl2va"].available
            else explicit_reference_video
        )
        reference_output = provided_outputs["reference_ref2va"]
        if reference_output is not None:
            reference_ref2va = provided_task(
                name="reference_ref2va",
                backend="legacy_pytorch",
                output_path=reference_output,
            )
        elif shared_reference_video is not None:
            reference_ref2va = run_task(
                name="reference_ref2va",
                backend="legacy_pytorch",
                command=build_command(
                    generator_script=generator_script,
                    task="ref2va",
                    model_path=model_path,
                    gpus=legacy_gpus,
                    output_path=generated_outputs["reference_ref2va"],
                    reference_video=shared_reference_video,
                    legacy=True,
                    legacy_text_gpus=args.legacy_text_gpus,
                    legacy_transformer_gpus=args.legacy_transformer_gpus,
                    legacy_memory_fraction=legacy_memory_fraction,
                ),
                output_path=generated_outputs["reference_ref2va"],
                log_path=logs["reference_ref2va"],
                cwd=repo_root,
            )
        else:
            reference_ref2va = skipped_task(
                name="reference_ref2va",
                backend="legacy_pytorch",
                output_path=generated_outputs["reference_ref2va"],
                log_path=logs["reference_ref2va"],
                reason="the shared Ref2VA input video is unavailable",
            )
        task_results["reference_ref2va"] = reference_ref2va

    if "fl2va" in selected_tasks:
        infinilm_output = provided_outputs["infinilm_fl2va"]
        if infinilm_output is not None:
            infinilm_fl2va = provided_task(
                name="infinilm_fl2va",
                backend="infinilm",
                output_path=infinilm_output,
            )
        else:
            assert first_frame is not None
            infinilm_fl2va = run_task(
                name="infinilm_fl2va",
                backend="infinilm",
                command=build_command(
                    generator_script=generator_script,
                    task="fl2va",
                    model_path=model_path,
                    gpus=args.infinilm_gpus,
                    output_path=generated_outputs["infinilm_fl2va"],
                    first_frame=first_frame,
                    legacy=False,
                ),
                output_path=generated_outputs["infinilm_fl2va"],
                log_path=logs["infinilm_fl2va"],
                cwd=repo_root,
            )
        task_results["infinilm_fl2va"] = infinilm_fl2va

    if "ref2va" in selected_tasks:
        infinilm_output = provided_outputs["infinilm_ref2va"]
        if infinilm_output is not None:
            infinilm_ref2va = provided_task(
                name="infinilm_ref2va",
                backend="infinilm",
                output_path=infinilm_output,
            )
        elif shared_reference_video is not None:
            infinilm_ref2va = run_task(
                name="infinilm_ref2va",
                backend="infinilm",
                command=build_command(
                    generator_script=generator_script,
                    task="ref2va",
                    model_path=model_path,
                    gpus=args.infinilm_gpus,
                    output_path=generated_outputs["infinilm_ref2va"],
                    reference_video=shared_reference_video,
                    legacy=False,
                ),
                output_path=generated_outputs["infinilm_ref2va"],
                log_path=logs["infinilm_ref2va"],
                cwd=repo_root,
            )
        else:
            infinilm_ref2va = skipped_task(
                name="infinilm_ref2va",
                backend="infinilm",
                output_path=generated_outputs["infinilm_ref2va"],
                log_path=logs["infinilm_ref2va"],
                reason="the shared Ref2VA input video is unavailable",
            )
        task_results["infinilm_ref2va"] = infinilm_ref2va

    comparisons: dict[str, dict[str, Any]] = {}
    for task in selected_tasks:
        reference = task_results[f"reference_{task}"]
        infinilm = task_results[f"infinilm_{task}"]
        comparisons[task] = compare_task_outputs(task.upper(), reference, infinilm)
        print_timing_comparison(task.upper(), reference, infinilm)

    print_final_report(comparisons)
    if any(not result.available for result in task_results.values()):
        return 2
    if any(result["status"] != "complete" for result in comparisons.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
