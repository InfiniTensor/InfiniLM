#!/usr/bin/env python3
"""
一键对比 chunked prefill 开/关性能

该脚本会依次启动 launch_server.py (chunk-size=0/256)，运行 test_perf_mix.py 取结果，最后输出对比。
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LM_DIR = os.path.dirname(SCRIPT_DIR)
INFERENCE_SERVER = os.path.join(LM_DIR, "python", "infinilm", "server", "inference_server.py")
TEST_SCRIPT = os.path.join(SCRIPT_DIR, "test_perf_mix.py")


from openai import OpenAI, APIConnectionError, APIStatusError

def wait_for_server(popen, host, port, model, timeout=300):
    client = OpenAI(base_url=f"http://{host}:{port}", api_key="default")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if popen.poll() is not None:
            raise RuntimeError(f"server exited early with code {popen.returncode}")
        try:
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            return
        except (APIConnectionError, APIStatusError):
            time.sleep(1)
    raise TimeoutError(f"server not ready within {timeout}s")


def inference_server(chunk_size, device, port, batch_size, max_new_tokens, enable_paged_attn, enable_graph, model_path):
    print(INFERENCE_SERVER)
    args = ["CUDA_VISIBLE_DEVICES=8", sys.executable, INFERENCE_SERVER,
            f"--chunk-size {chunk_size}",
            f"--device {device}",
            f"--port {port}",
            f"--batch-size {batch_size}",
            f"--max-new-tokens {max_new_tokens}",
            f"--model {model_path}",
            "--enable-chunk-prefill-graph"]
    if enable_paged_attn:
        args.append("--enable-paged-attn")
    if enable_graph:
        args.append("--enable-graph")

    popen = subprocess.Popen(" ".join(args), shell=True, preexec_fn=os.setsid, stderr=subprocess.STDOUT)
    return popen


import socket

def stop_server(popen, host="127.0.0.1", port=2333, timeout=30):
    if popen and popen.poll() is None:
        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
        try:
            popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
            popen.wait(timeout=5)

    # 等端口真正释放（uvicorn 在 graceful shutdown 期间端口还开着）
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass  # 还有人监听，继续等
        except OSError:
            return   # 端口已释放
        time.sleep(0.3)
    raise RuntimeError(f"port {port} still in use after stop_server")


def run_test_perf():
    cmd = f"{sys.executable} -u {TEST_SCRIPT}"
    proc = subprocess.Popen(
        cmd, shell=True, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush() # 子终端的输出直接转发到父终端，保持实时显示
        lines.append(line)
    proc.wait()
    return proc.returncode, "".join(lines)   # 返回码 + test_perf_mix.py的输出文本

def parse_stats(output):
    def grab(pat):
        m = re.search(pat, output)
        return float(m.group(1)) if m else None
    
    success_m = re.search(r"成功请求数\s*:\s*(\d+)", output)
    return {
        "avg_ttft_s":       grab(r"Average TTFT\s*:\s*([0-9.]+)\s*s"),
        "avg_e2e_s":        grab(r"Average latency\s*:\s*([0-9.]+)\s*s"),
        "avg_ms_per_token": grab(r"Avg time per token\s*:\s*([0-9.]+)\s*ms/token"),
        "avg_tps":          grab(r"Avg Token generation speed\s*:\s*([0-9.]+)"),
        "rps":              grab(r"请求速率 \(RPS\)\s*:\s*([0-9.]+)"),
        "success":          int(success_m.group(1)) if success_m else None,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比较 chunked prefill 开/关的 TTFT/E2E")
    parser.add_argument("--device", type=str, default="iluvatar", help="设备类型")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--enable-paged-attn", type=bool, default=True)
    parser.add_argument("--enable-graph", type=bool, default=True)
    parser.add_argument("--port", type=int, default=2333)
    parser.add_argument("--model-path", type=str, default="/data-aisoft/mechdancer/models/9g_8b_thinking_llama/")

    
    args = parser.parse_args()

    results = []

    for chunk_size in (0, 256):
        mode = "ON" if chunk_size > 0 else "OFF"
        print("\n" + "="*78)
        print(f"开始部署大模型推理服务，chunked prefill {mode} (chunk-size={chunk_size})，请等待服务启动完成...")

        server = inference_server(chunk_size=chunk_size, device=args.device, port=args.port,
                                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                                enable_paged_attn=args.enable_paged_attn, enable_graph=args.enable_graph,
                                model_path=args.model_path)
        try:
            wait_for_server(server, "127.0.0.1", args.port, model="FM9G-7B", timeout=300)
            print("服务启动完成，开始跑 test_perf_mix.py (上一条200OK请求为服务测试成功标志)")
            retcode, out = run_test_perf()  # out为test子终端输出文本
            if retcode != 0:
                print("test_perf_cp.py 执行失败，退出码", retcode)
                print(out)
                raise SystemExit(1)
            
            stats = parse_stats(out)  #从test输出文本中提取性能指标
            stats.update({"chunk_size": chunk_size, "mode": mode})  
            results.append(stats)  
            print(f"完成chunked prefill {mode}测试 -> {stats}")

        finally:
            stop_server(server, host="127.0.0.1", port=args.port)
            print("服务已停止")

    print("\n" + "#"*78)
    print("最终对比（chunked prefill ON vs OFF）")
    print("-"*78)

    def fmt(v, unit="", spec=".3f"):
        return "N/A" if v is None else f"{v:{spec}}{unit}"

    def diff(a, b):
        return None if (a is None or b is None) else a - b

    def speedup_pct(on, off):
        # 越小越快的指标：正数 = ON 比 OFF 快
        if on is None or off is None or off == 0:
            return None
        return (off - on) / off * 100

    on_r  = next((r for r in results if r["mode"] == "ON"),  None)
    off_r = next((r for r in results if r["mode"] == "OFF"), None)

    print(f"{'指标':<22}{'ON':>14}{'OFF':>14}{'Δ (ON-OFF)':>16}{'ON 提升':>12}")
    print("-"*78)

    def row(label, key, unit, spec=".3f", lower_is_better=True):
        a = (on_r  or {}).get(key)
        b = (off_r or {}).get(key)
        pct = speedup_pct(a, b) if lower_is_better else speedup_pct(b, a)
        print(
            f"{label:<22}"
            f"{fmt(a, unit, spec):>14}"
            f"{fmt(b, unit, spec):>14}"
            f"{fmt(diff(a, b), unit, '+'+spec):>16}"
            f"{fmt(pct, '%', '+.2f'):>12}"
        )

    row("Avg TTFT",        "avg_ttft_s",       " s")
    row("Avg E2E latency", "avg_e2e_s",        " s")
    row("Avg ms/token",    "avg_ms_per_token", " ms", ".2f")
    row("Avg tokens/s",    "avg_tps",          "",    ".2f", lower_is_better=False)
    row("RPS",             "rps",              "",    ".2f", lower_is_better=False)
    print("-"*78)
