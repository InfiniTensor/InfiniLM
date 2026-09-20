"""Shared workload definition for the offline dual-engine perf baseline.

Both bench entry points (infinilm / vllm) import from here so the workload
is byte-identical across engines. Stdlib + huggingface_hub only.
"""

import os
import subprocess
import threading
import time

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SHORT_PROMPTS = [
    "如果猫能写诗，它们会写些什么？",
    "描述一个没有重力的世界。",
    "如果地球停止自转，会发生什么？",
    "假设你是一只会飞的鲸鱼，描述你的日常生活。",
]

_BASE_PARA = (
    "大语言模型推理引擎的核心挑战在于如何高效利用 GPU 的算力与显存带宽。"
    "预填充阶段是计算密集型的，而解码阶段则主要受限于显存带宽，"
    "因为每一步都需要读取全部的模型权重与不断增长的 KV 缓存。"
    "连续的批处理调度与前缀缓存是提升服务端吞吐的关键技术。"
)

LONG_PROMPT = _BASE_PARA * 40  # ~2k tokens; actual count recorded at runtime


def _batch_prompts(n: int) -> list[str]:
    return [f"问题 {i + 1}：{SHORT_PROMPTS[i % len(SHORT_PROMPTS)]}" for i in range(n)]


def concurrent_prefill_workload(n: int = 8) -> tuple:
    """w5: n long prompts (w2's LONG_PROMPT construction) submitted in one
    batch — the chunked-prefill acceptance workload. With plain FCFS
    scheduling the n-1 queued prefills block everyone's decode; chunked
    prefill should interleave them and cut e2e wall time.

    Each prompt gets a distinct 问题 i： header so they diverge from the
    very first tokens — prefix caching (on for both engines) can then
    never dedupe them, and every request pays its own full prefill.
    """
    prompts = [f"问题 {i + 1}：{LONG_PROMPT}" for i in range(n)]
    return ("w5_concurrent_prefill", prompts, 128)


def decode_stall_workload(n_decode: int = 8, inject_reps: int = 80) -> tuple:
    """w6: the chunked-prefill *benefit* scenario — a steady decode stream
    (n_decode short prompts, long generation) with one long prompt injected
    mid-flight. Plain FCFS stalls every decode token for the whole injected
    prefill; chunked prefill should spread it into mixed steps and shrink the
    ITL spike. bench.py drives this engine-level (add_request/step); the
    prompts/max_tokens fields are placeholders, parameters ride in the tuple.

    The injected prompt gets a unique nonce header so prefix caching (w2/w5
    ran earlier in the same engine) can never dedupe its blocks: the nonce
    shifts every 256-token block boundary vs any earlier LONG_PROMPT use.
    """
    inject_prompt = f"w6注入{time.time_ns()}：" + _BASE_PARA * inject_reps
    return ("w6_decode_stall", (n_decode, inject_prompt), 0)


def repetitive_copy_workload() -> tuple:
    """w7: prompt-lookup 投机采样的收益负载——pattern 续写使输出大段
    重复 prompt 内容，n-gram 后缀匹配持续命中（接受率上限的演示）。
    用纯模式重复而非指令（"请重复 N 遍"），贪心小模型必然续写，
    不依赖指令遵循能力。正确性由 verify 保证，与是否命中无关。
    """
    prompt = "下面这段文字会不断重复：\n" + _BASE_PARA * 3
    return ("w7_repetitive_copy", [prompt], 512)


# name, prompts, max_tokens
WORKLOADS = [
    ("w1_short_decode", SHORT_PROMPTS[:1], 128),
    ("w2_long_prefill", [LONG_PROMPT], 128),
    ("w3_batch32", _batch_prompts(32), 128),
    ("w4_long_decode", SHORT_PROMPTS[1:2], 1024),
    concurrent_prefill_workload(),
    decode_stall_workload(),
    repetitive_copy_workload(),
]


def ctx_sweep_workloads() -> list:
    """Single-request decode-vs-context-length sweep for the decode-kernel
    crossover study (v9/v11: paged splitkv wins short ctx, FA kvcache wins
    long ctx; crossover is model-geometry dependent).

    `_BASE_PARA` is ~81 tokens/repeat on the Qwen3 tokenizer (40 reps
    tokenize to ~3240), so the multiplier list below spans ~0.2k~5k ctx.
    The label is only approximate — the exact prompt token count is recorded
    in the report at runtime. Prefill is identical across the two backends
    (both use FA2), so the e2e delta at each ctx isolates the decode kernel.
    """
    return [(f"ctx81x{k}", [_BASE_PARA * k], 128) for k in (2, 4, 8, 12, 16, 24, 32, 40, 48, 64)]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_model_path(name_or_path: str) -> str:
    """Accept a local dir or an HF repo id; return a local directory path."""
    if os.path.isdir(name_or_path):
        return os.path.abspath(name_or_path)
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)


class MemSampler(threading.Thread):
    """Sample `nvidia-smi` memory.used in the background; track peak."""

    def __init__(self, interval: float = 0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.latest = 0
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                ).strip()
                self.latest = int(out.splitlines()[0])
                self.peak = max(self.peak, self.latest)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=2)
