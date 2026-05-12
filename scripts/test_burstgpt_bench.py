#!/usr/bin/env python3
"""一键对比 optimized 开/关时的 BurstGPT 服务压测结果。

该脚本会依次启动 InfiniLM 服务：
optimized OFF 和 ON ，
分别调用pure_bench_serve.py跑BurstGPT测试集，最后输出对比表。

BurstGPT dataset 只取第 2、3 列(Request tokens / Response tokens)做输入输出长度,没有读取时间戳列,到达时间用 request_rate=2.0 + burstiness=1.0 的 Poisson 过程发出
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import selectors
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SERVER = PROJECT_ROOT / "python" / "infinilm" / "server" / "inference_server.py"
BENCH_SCRIPT = Path(__file__).resolve().parent / "pure_bench_serve.py"
DEFAULT_MODEL = "/data-aisoft/mechdancer/models/9g_8b_thinking/"    # 改成机器上对应的模型路径
DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets/BurstGPT/BurstGPT_1000.csv"
DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "bench_results"
USE_COLOR = False

def paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def visual_width(text: str) -> int:
    width = 0
    for ch in text:
        width += 2 if unicodedata.east_asian_width(ch) in {"F", "W"} else 1
    return width


def ljust_display(text: str, width: int) -> str:
    return text + " " * max(0, width - visual_width(text))


def print_bar(title: str, fill: str = "=") -> None:
    line = fill * 78
    print("\n" + paint(line, "36;1"))
    print(paint(title, "36;1"))
    print(paint(line, "36;1"))


def print_kv(key: str, value: object) -> None:
    print(f"  {paint(ljust_display(key, 18), '2')}: {value}")


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分{sec:02d}秒"
    if minutes:
        return f"{minutes}分{sec:02d}秒"
    return f"{sec}秒"


def terminate_process_group(popen: subprocess.Popen, timeout: int = 30) -> None:
    if popen.poll() is not None:
        return
    os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
    try:
        popen.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
        popen.wait(timeout=5)


def wait_for_port_free(host: str, port: int, timeout: int = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass
        except OSError:
            return
        time.sleep(0.3)
    raise RuntimeError(f"port {port} still in use after stop_server")


def parse_rate(value: str) -> float:
    if value.lower() in {"inf", "infinity"}:
        return float("inf")
    return float(value)


def rate_to_str(value: float) -> str:
    return "inf" if math.isinf(value) else str(value)


def sanitize(value: str) -> str:
    return value.replace("/", "_").replace(".", "p").replace("-", "m")


def count_and_filter_dataset(args: argparse.Namespace) -> Path:
    print_bar("准备 BurstGPT 数据")
    src = Path(args.dataset_path)
    if not src.exists():
        raise FileNotFoundError(f"BurstGPT dataset not found: {src}")

    print_kv("原始数据集", src)
    if args.use_full_dataset:
        print("  使用完整 CSV，不做长度过滤。")
        return src

    out = Path(args.filtered_dataset_path) if args.filtered_dataset_path else None
    if out is None:
        out = src.with_name(
            f"{src.stem}_gpt4_pos_req{args.max_request_tokens}_resp{args.max_response_tokens}.csv"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    print_kv("过滤后数据集", out)
    print_kv("过滤条件", f"Model=GPT-4, 0 < 输出 token <= {args.max_response_tokens}, 输入 token <= {args.max_request_tokens}")

    total = gpt4 = positive = kept = 0
    with src.open(newline="", encoding="utf-8") as f, out.open(
        "w", newline="", encoding="utf-8"
    ) as g:
        reader = csv.DictReader(f)
        writer = csv.DictWriter(g, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total += 1
            if row.get("Model") != "GPT-4":
                continue
            gpt4 += 1
            try:
                req_tokens = int(float(row.get("Request tokens") or 0))
                resp_tokens = int(float(row.get("Response tokens") or 0))
            except ValueError:
                continue
            if resp_tokens <= 0:
                continue
            positive += 1
            if req_tokens > args.max_request_tokens or resp_tokens > args.max_response_tokens:
                continue
            writer.writerow(row)
            kept += 1

    print_kv("原始总行数", total)
    print_kv("GPT-4 行数", gpt4)
    print_kv("GPT-4 且输出非空", positive)
    print_kv("本次可用行数", kept)
    if kept == 0:
        raise RuntimeError("Filtered BurstGPT dataset is empty; relax token limits.")
    if kept < args.num_prompts:
        print(
            f"  注意：过滤后只有 {kept} 条，但本次请求 {args.num_prompts} 条；"
            "vLLM 会重复采样。"
        )
    return out


def chat_warmup(base_url: str, model: str, timeout: int = 30) -> None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if not 200 <= resp.status < 300:
            raise RuntimeError(f"warmup request failed with HTTP {resp.status}")


def wait_for_server(
    base_url: str,
    popen: subprocess.Popen | None,
    timeout: int,
    model: str,
) -> None:
    deadline = time.time() + timeout
    started = time.monotonic()
    last_notice = 0.0
    health_url = f"{base_url}/health"
    print_kv("健康检查", health_url)
    while time.time() < deadline:
        if popen is not None and popen.poll() is not None:
            raise RuntimeError(f"server exited early with code {popen.returncode}")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    chat_warmup(base_url, model)
                    print(f"  服务已就绪，用时 {format_duration(time.monotonic() - started)}。")
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        now = time.monotonic()
        if now - last_notice >= 15:
            print(f"  正在等待服务启动... 已等待 {format_duration(now - started)}")
            last_notice = now
        time.sleep(2)
    raise TimeoutError(f"server not ready within {timeout}s: {health_url}")


def start_server(
    args: argparse.Namespace,
    chunk_size: int,
    mode_label: str,
) -> subprocess.Popen | None:
    base_url = f"http://{args.client_host}:{args.port}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    cmd = [
        sys.executable,
        str(INFERENCE_SERVER),
        "--device",
        args.device,
        "--model",
        args.model_path,
        "--backend",
        args.backend,
        "--max-batch-size",
        str(args.max_batch_size),
        "--max-new-tokens",
        str(args.server_max_new_tokens),
        "--host",
        args.server_host,
        "--port",
        str(args.port),
    ]
    if args.enable_paged_attn:
        cmd.append("--enable-paged-attn")
    if args.enable_graph:
        cmd.append("--enable-graph")
    if args.enable_chunk_prefill_graph:
        cmd.append("--enable-chunk-prefill-graph")
    cmd.extend(["--chunk-size", str(chunk_size)])

    print_bar(
        f"开始部署大模型推理服务，optimized {mode_label} "
        f"(chunk-size={chunk_size})，请等待服务启动完成..."
    )
    print_kv("GPU", f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
    print_kv("模型", args.model_path)
    print_kv("后端", args.backend)
    print_kv("optimized", mode_label)
    if args.show_server_output:
        print_kv("服务输出", "直接显示在终端")

    server_stdout = None if args.show_server_output else subprocess.DEVNULL
    popen = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=server_stdout,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )
    try:
        wait_for_server(base_url, popen, args.server_timeout, args.served_model_name)
    except Exception:
        print("\n服务启动失败。需要看服务端输出时，可以加 --show-server-output 重新运行。")
        raise
    return popen


def stop_server(
    popen: subprocess.Popen | None,
    host: str = "127.0.0.1",
    port: int = 2333,
    timeout: int = 30,
) -> None:
    if popen is None or popen.poll() is not None:
        wait_for_port_free(host, port, timeout)
        return
    terminate_process_group(popen, timeout)
    wait_for_port_free(host, port, timeout)


def should_show_vllm_line(line: str) -> bool:
    # 正常指标由 print_summary 统一用中文输出；这里只透出明显异常。
    text = line.strip()
    if not text:
        return False
    error_prefixes = (
        "ERROR",
        "CRITICAL",
        "Traceback",
    )
    error_snippets = (
        "RuntimeError",
        "ValueError",
        "ConnectionError",
        "CUDA out of memory",
        "No such file",
    )
    return text.startswith(error_prefixes) or any(snippet in text for snippet in error_snippets)


def parse_progress(line: str) -> int | None:
    # 解析 pure_bench_serve.py 打出的 "PROGRESS x/n" 进度行，返回已完成数 x。
    marker = "PROGRESS "
    idx = line.find(marker)
    if idx == -1:
        return None
    frag = line[idx + len(marker):].strip().split()
    if not frag or "/" not in frag[0]:
        return None
    try:
        return int(frag[0].split("/")[0])
    except ValueError:
        return None


def run_command_with_heartbeat(
    cmd: list[str],
    cwd: Path,
    args: argparse.Namespace,
    result_path: Path,
) -> None:
    started = time.monotonic()
    last_heartbeat = started
    captured: list[str] = []
    completed_count = 0

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid,
    )
    selector = selectors.DefaultSelector()
    if proc.stdout is not None:
        selector.register(proc.stdout, selectors.EVENT_READ)

    try:
        while proc.poll() is None:
            for key, _ in selector.select(timeout=1):
                line = key.fileobj.readline()
                if not line:
                    continue
                captured.append(line)
                parsed = parse_progress(line)
                if parsed is not None:
                    completed_count = parsed
                elif args.show_vllm_output or should_show_vllm_line(line):
                    print("  " + line.rstrip())

            now = time.monotonic()
            if args.progress_interval > 0 and now - last_heartbeat >= args.progress_interval:
                print(
                    "  压测正在运行："
                    f"已用 {format_duration(now - started)}，"
                    f"目标 {args.num_prompts} 条，已完成 {completed_count} 条，"
                    f"并发上限 {args.max_concurrency}。"
                )
                last_heartbeat = now

        if proc.stdout is not None:
            for line in proc.stdout:
                captured.append(line)
                if args.show_vllm_output or should_show_vllm_line(line):
                    print("  " + line.rstrip())
    except KeyboardInterrupt:
        print("\n收到中止信号，正在停止 vLLM benchmark 子进程...")
        terminate_process_group(proc, timeout=10)
        raise
    finally:
        selector.close()

    if proc.returncode != 0:
        print("\nvLLM benchmark 异常退出，最近输出如下：")
        meaningful = [ln for ln in captured if parse_progress(ln) is None]
        for line in meaningful[-80:]:
            print("  " + line.rstrip())
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_benchmark(
    args: argparse.Namespace,
    dataset_path: Path,
    result_path: Path,
    mode_label: str,
) -> None:
    cmd = [
        sys.executable,
        str(BENCH_SCRIPT),
        "--backend",
        "openai-chat",
        "--base-url",
        f"http://{args.client_host}:{args.port}",
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        args.served_model_name,
        "--tokenizer",
        args.tokenizer or args.model_path,
        "--dataset-name",
        "burstgpt",
        "--dataset-path",
        str(dataset_path),
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        rate_to_str(args.request_rate),
        "--seed",
        str(args.seed),
        "--burstiness",
        str(args.burstiness),
        "--max-concurrency",
        str(args.max_concurrency),
        "--disable-tqdm",
        "--save-result",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    ]
    if args.save_detailed:
        cmd.append("--save-detailed")
    if args.temperature is not None:
        cmd.extend(["--temperature", str(args.temperature)])
    if args.ignore_eos:
        cmd.append("--ignore-eos")

    print(paint(f"服务启动完成，开始跑 BurstGPT / vLLM benchmark（optimized {mode_label}）", "32;1"))
    print_kv("请求数", args.num_prompts)
    print_kv("请求速率", rate_to_str(args.request_rate))
    print_kv("并发上限", args.max_concurrency)
    print_kv("随机种子", args.seed)
    print_kv("数据集", dataset_path)
    if args.save_result_files:
        print_kv("结果文件", result_path)
    else:
        print_kv("结果文件", "不保留（使用临时文件读取结果）")
    print_kv("原始输出", "默认隐藏；需要调试时加 --show-vllm-output")
    print("  开始压测。这一步可能持续较久，脚本会定时打印运行状态。")
    run_command_with_heartbeat(cmd, PROJECT_ROOT, args, result_path)


def load_result(result_path: Path) -> dict:
    with result_path.open(encoding="utf-8") as f:
        return json.load(f)


def fmt(value: object, unit: str = "", spec: str = ".3f") -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:{spec}}{unit}"
    return str(value)


def print_run_summary(stats: dict) -> None:
    print("本轮结果：")
    if stats.get("result_path"):
        print_kv("结果文件", stats["result_path"])
    print_kv("完成请求数", fmt(stats.get("completed"), "", ".0f"))
    print_kv("失败请求数", fmt(stats.get("failed"), "", ".0f"))
    print_kv("请求吞吐", fmt(stats.get("request_throughput"), " req/s", ".3f"))
    print_kv("输出吞吐", fmt(stats.get("output_throughput"), " tok/s", ".3f"))
    print_kv("平均 TTFT", fmt(stats.get("mean_ttft_ms"), " ms", ".2f"))
    print_kv("平均 TPOT", fmt(stats.get("mean_tpot_ms"), " ms", ".2f"))


def diff(on_value: object, off_value: object) -> float | None:
    if not isinstance(on_value, (int, float)) or not isinstance(off_value, (int, float)):
        return None
    return on_value - off_value


def speedup_pct(on_value: object, off_value: object, lower_is_better: bool) -> float | None:
    if not isinstance(on_value, (int, float)) or not isinstance(off_value, (int, float)):
        return None
    if off_value == 0:
        return None
    if lower_is_better:
        return (off_value - on_value) / off_value * 100
    return (on_value - off_value) / off_value * 100


def print_comparison(results: list[dict]) -> None:
    print("\n" + paint("#" * 78, "35;1"))
    print(paint("最终对比（optimized ON vs OFF）", "35;1"))
    print(paint("-" * 78, "35;1"))

    on_r = next((r for r in results if r["mode"] == "ON"), None)
    off_r = next((r for r in results if r["mode"] == "OFF"), None)
    if not on_r or not off_r:
        only = results[0]
        print(f"只跑了 optimized {only['mode']}，没有生成 ON/OFF 对比。")
        print_run_summary(only)
        return

    header = (
        f"{ljust_display('指标', 22)}"
        f"{'ON':>14}"
        f"{'OFF':>14}"
        f"{'Δ (ON-OFF)':>16}"
        f"{'ON 提升':>12}"
    )
    print(paint(header, "1"))
    print("-" * 78)

    def row(label: str, key: str, unit: str, spec: str = ".3f", lower_is_better: bool = True) -> None:
        on_value = on_r.get(key)
        off_value = off_r.get(key)
        delta = diff(on_value, off_value)
        pct = speedup_pct(on_value, off_value, lower_is_better)
        pct_text = f"{fmt(pct, '%', '+.2f'):>12}"
        if isinstance(pct, (int, float)):
            pct_text = paint(pct_text, "32;1" if pct >= 0 else "31;1")
        print(
            f"{ljust_display(label, 22)}"
            f"{fmt(on_value, unit, spec):>14}"
            f"{fmt(off_value, unit, spec):>14}"
            f"{fmt(delta, unit, '+' + spec):>16}"
            f"{pct_text}"
        )

    row("完成请求数", "completed", "", ".0f", lower_is_better=False)
    row("失败请求数", "failed", "", ".0f", lower_is_better=True)
    row("总耗时", "duration", " s", ".2f", lower_is_better=True)
    row("请求吞吐", "request_throughput", " req/s", ".3f", lower_is_better=False)
    row("输出吞吐", "output_throughput", " tok/s", ".3f", lower_is_better=False)
    row("Avg TTFT", "mean_ttft_ms", " ms", ".2f", lower_is_better=True)
    row("Median TTFT", "median_ttft_ms", " ms", ".2f", lower_is_better=True)
    row("P99 TTFT", "p99_ttft_ms", " ms", ".2f", lower_is_better=True)
    row("Avg TPOT", "mean_tpot_ms", " ms", ".2f", lower_is_better=True)
    row("Median TPOT", "median_tpot_ms", " ms", ".2f", lower_is_better=True)
    row("P99 TPOT", "p99_tpot_ms", " ms", ".2f", lower_is_better=True)
    print("-" * 78)
    if on_r.get("result_path"):
        print_kv("ON 结果文件", on_r["result_path"])
    if off_r.get("result_path"):
        print_kv("OFF 结果文件", off_r["result_path"])


def benchmark_modes(args: argparse.Namespace) -> list[tuple[str, int]]:
    if args.modes == "off":
        return [("OFF", 0)]
    if args.modes == "on":
        return [("ON", args.chunk_size)]
    return [("OFF", 0), ("ON", args.chunk_size)]


def with_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def default_result_path(args: argparse.Namespace, result_dir: Path) -> Path:
    rate = sanitize(rate_to_str(args.request_rate))
    return result_dir / (
        f"vllm_burstgpt_{args.num_prompts}req_rps{rate}_mc{args.max_concurrency}.json"
    )


def result_path_for_mode(args: argparse.Namespace, result_dir: Path, mode_label: str) -> Path:
    base = result_dir / args.result_filename if args.result_filename else default_result_path(args, result_dir)
    return with_suffix(base, f"chunk_{mode_label.lower()}")


def validate_args(args: argparse.Namespace, modes: list[tuple[str, int]]) -> None:
    if args.num_prompts <= 0:
        raise SystemExit("--num-prompts 必须大于 0。")
    if args.max_concurrency <= 0:
        raise SystemExit("--max-concurrency 必须大于 0。")
    if args.max_request_tokens <= 0 or args.max_response_tokens <= 0:
        raise SystemExit("--max-request-tokens / --max-response-tokens 必须大于 0。")
    if any(mode == "ON" for mode, _ in modes) and args.chunk_size <= 0:
        raise SystemExit("optimized ON 需要 --chunk-size > 0。")
    if args.result_filename and not args.save_result_files:
        print("  注意：未加 --save-result-files，--result-filename 只用于临时文件名，不会保留。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对比 optimized 开/关时的 BurstGPT / vLLM benchmark serve 结果。"
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--filtered-dataset-path", default=None)
    parser.add_argument("--use-full-dataset", action="store_true")
    parser.add_argument("--max-request-tokens", type=int, default=1024)
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--request-rate", type=parse_rate, default=2.0) # 平均速率默认 2 req/s
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--result-filename", default=None)
    parser.add_argument("--save-result-files", action="store_true", help="保留 vLLM JSON 结果文件；默认只在临时文件中读取结果，跑完不保留。")
    parser.add_argument("--save-detailed", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "11"))
    parser.add_argument("--device", default="iluvatar")  # 换成自己的设备类型
    parser.add_argument("--backend", default="cpp", choices=["cpp", "python", "torch", "vllm"])
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--client-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2333)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--server-max-new-tokens", type=int, default=256)
    parser.add_argument("--enable-paged-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-chunk-prefill-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-size", type=int, default=256, help="optimized ON 时使用的 chunk-size；OFF 固定为 0。")
    parser.add_argument("--modes", choices=["both", "off", "on"], default="both", help="默认 both：先 OFF 后 ON 并输出最终对比。")
    parser.add_argument("--server-timeout", type=int, default=600)
    parser.add_argument("--served-model-name", default="9g_8b_thinking")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=10, help="压测运行中的中文心跳提示间隔，单位秒；设为 0 可关闭。")
    parser.add_argument("--show-vllm-output", action="store_true", help="显示 benchmark 的原始英文输出。")
    parser.add_argument("--show-server-output", action="store_true", help="直接显示 InfiniLM 服务端输出；默认隐藏且不保存日志。")
    parser.add_argument("--color", choices=["auto", "always", "never"], default="auto", help="终端颜色输出。")
    return parser.parse_args()


def main() -> None:
    global USE_COLOR
    args = parse_args()
    USE_COLOR = (
        args.color == "always"
        or (args.color == "auto" and sys.stdout.isatty() and not os.environ.get("NO_COLOR"))
    )

    modes = benchmark_modes(args)
    validate_args(args, modes)

    result_dir = Path(args.result_dir)
    if args.save_result_files:
        result_dir.mkdir(parents=True, exist_ok=True)
    print_bar("课题1：高性能统一智能计算架构及编译优化技术")
    print_bar("课题1.3：负载资源互感知编译优化")
    print_bar("优化技术效果对比测试")
    print_kv("模型目录", args.model_path)
    print_kv("请求数", args.num_prompts)
    print_kv("请求速率", rate_to_str(args.request_rate))
    print_kv("并发上限", args.max_concurrency)
    print_kv("GPU", args.cuda_visible_devices)
    print_kv("对比模式", " -> ".join(mode for mode, _ in modes))
    print_kv("文件保存", "保存 JSON 结果" if args.save_result_files else "不保存日志和结果文件")


    dataset_path = count_and_filter_dataset(args)
    results: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="infinilm-burstgpt-") as tmpdir:
        run_result_dir = result_dir if args.save_result_files else Path(tmpdir)

        for mode_label, chunk_size in modes:
            result_path = result_path_for_mode(args, run_result_dir, mode_label)
            popen: subprocess.Popen | None = None
            try:
                popen = start_server(args, chunk_size, mode_label)
                run_benchmark(args, dataset_path, result_path, mode_label)
                stats = load_result(result_path)
                stats.update(
                    {
                        "mode": mode_label,
                        "chunk_size": chunk_size,
                        "result_path": str(result_path) if args.save_result_files else "",
                    }
                )
                results.append(stats)
                print_run_summary(stats)
                print(
                    paint(
                        f"完成 optimized {mode_label} 测试 -> "
                        f"成功 {fmt(stats.get('completed'), '', '.0f')}，"
                        f"失败 {fmt(stats.get('failed'), '', '.0f')}，"
                        f"请求吞吐 {fmt(stats.get('request_throughput'), ' req/s', '.3f')}，"
                        f"平均 TTFT {fmt(stats.get('mean_ttft_ms'), ' ms', '.2f')}",
                        "32;1",
                    )
                )
            finally:
                stop_server(popen, host=args.client_host, port=args.port)
                print("服务已停止")

    print_comparison(results)


if __name__ == "__main__":
    main()
