"""Chunked prefill + prefill/decode 混排的离线冒烟测试（无需 GPU）。

infinilm 包本体依赖 infinicore/torch/transformers/janus 等重型库，本测试通过
sys.modules 预置轻量 stub，只真实加载纯 Python 的 scheduler / cache_manager /
request / basic_llm_processor / llm 模块，验证：

1. chunked 模式下 decode 优先、prefill 按剩余预算切块，二者可混入同一批次；
2. 混排批次构建的 input_ids/position_ids/past_kv_lengths/total_kv_lengths/
   cu_seqlens/input_offsets/slot_mapping 均按 (num_cached, chunk_len) 切片；
3. 中段 chunk 请求不 append 废 token、只推进 num_computed_tokens 并逐 chunk
   发布 prefix-cache 块；
4. 多模态请求不切分，保持整段 prefill；
5. 开关关闭时（旧模式）调度与输入构建行为与旧逻辑一致。

直接运行: python3 test/test_chunked_prefill.py
"""

import collections
import hashlib
import queue
import struct
import sys
import types
from pathlib import Path

REPO_PYTHON = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(REPO_PYTHON))


def _install_stubs():
    """预置轻量 stub，绕过 infinicore/torch/transformers/janus 等重型依赖。"""
    # 1) 包占位：避免执行各 __init__.py（会链入 torch/infinicore）
    pkg_root = REPO_PYTHON / "infinilm"
    for name, rel in [
        ("infinilm", ""),
        ("infinilm.llm", "llm"),
        ("infinilm.llm.model_runner", "llm/model_runner"),
        ("infinilm.processors", "processors"),
        ("infinilm.config", "config"),
        ("infinilm.multimodal", "multimodal"),
    ]:
        mod = types.ModuleType(name)
        mod.__path__ = [str(pkg_root / rel) if rel else str(pkg_root)]
        sys.modules[name] = mod

    # 2) janus：调度器只用到 sync_q 的阻塞队列接口
    janus = types.ModuleType("janus")

    class _SyncQueue:
        def __init__(self):
            self._q = collections.deque()

        def put(self, item, block=True, timeout=None):
            self._q.append(item)

        put_nowait = put

        def get(self, block=True, timeout=None):
            if not self._q:
                raise queue.Empty
            return self._q.popleft()

        def get_nowait(self):
            return self.get()

        def qsize(self):
            return len(self._q)

        def empty(self):
            return not self._q

    class _JanusQueue:
        def __init__(self, maxsize=0):
            self.sync_q = _SyncQueue()
            self.async_q = None

    janus.Queue = _JanusQueue
    sys.modules["janus"] = janus

    # 3) numpy / xxhash：prefix_cache 只要求确定性的 16 字节哈希，用标准库模拟
    numpy = types.ModuleType("numpy")

    class _FakeArray:
        def __init__(self, token_ids):
            self._token_ids = list(token_ids)

        def __len__(self):
            return len(self._token_ids)

        def tobytes(self):
            return struct.pack(f"<{len(self._token_ids)}i", *self._token_ids)

    numpy.asarray = lambda token_ids, dtype=None: _FakeArray(token_ids)
    sys.modules["numpy"] = numpy

    xxhash = types.ModuleType("xxhash")

    class _Hasher:
        def __init__(self):
            self._buf = bytearray()

        def update(self, data):
            self._buf += data

        def digest(self):
            return hashlib.sha256(bytes(self._buf)).digest()[:16]

    xxhash.xxh3_128 = _Hasher
    sys.modules["xxhash"] = xxhash

    # 4) transformers：basic_llm_processor 顶层 import 用到
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    transformers.AutoProcessor = object
    sys.modules["transformers"] = transformers

    # 5) infinicore：builder 内部 import，from_list 只做数据捕获
    infinicore = types.ModuleType("infinicore")
    infinicore.int64 = "int64"
    infinicore.int32 = "int32"

    class _FakeTensor:
        def __init__(self, data, dtype):
            self.data = data
            self.dtype = dtype

    infinicore.from_list = lambda data, dtype=None: _FakeTensor(data, dtype)
    sys.modules["infinicore"] = infinicore

    # 6) llm.py 的重型依赖（只在引擎初始化时才真正使用）
    engine_config = types.ModuleType("infinilm.config.engine_config")
    engine_config.EngineConfig = object
    sys.modules["infinilm.config.engine_config"] = engine_config

    kv_transfer = types.ModuleType("infinilm.config.kv_transfer")
    kv_transfer.KVTransferConfig = object
    sys.modules["infinilm.config.kv_transfer"] = kv_transfer

    infer_engine = types.ModuleType("infinilm.infer_engine")
    infer_engine.model_uses_mamba_cache = lambda hf_config: False
    infer_engine.read_hf_config = lambda model_path: {}
    sys.modules["infinilm.infer_engine"] = infer_engine

    kv_connector = types.ModuleType("infinilm.kv_connector")
    kv_connector.KVConnectorFactory = object
    kv_connector.KVConnectorRole = object
    sys.modules["infinilm.kv_connector"] = kv_connector

    model_runner = types.ModuleType("infinilm.llm.model_runner.model_runner")
    model_runner.ModelRunner = object
    sys.modules["infinilm.llm.model_runner.model_runner"] = model_runner

    multimodal = types.ModuleType("infinilm.multimodal.multimodal")
    multimodal.resolve_multimodal_inputs = lambda messages: {}
    sys.modules["infinilm.multimodal.multimodal"] = multimodal


_install_stubs()

from infinilm.llm.llm import LLMEngine  # noqa: E402
from infinilm.llm.request import InferenceRequest  # noqa: E402
from infinilm.llm.sampling_params import SamplingParams  # noqa: E402
from infinilm.llm.scheduler import Scheduler  # noqa: E402
from infinilm.processors.basic_llm_processor import BasicLLMProcessor  # noqa: E402

_SAMPLED_TOKEN = 42  # 模拟 C++ 每请求采出的 token（非 eos）


def _make_request(req_id, prompt_token_ids, max_tokens=3, has_multimodal_inputs=False):
    return InferenceRequest(
        request_id=req_id,
        prompt="x" * len(prompt_token_ids),
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        eos_token_ids=[],
        has_multimodal_inputs=has_multimodal_inputs,
    )


def _make_engine(scheduler):
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = scheduler
    engine.eos_token_ids = []

    class _FakeTokenizer:
        def decode(self, token_ids):
            return ""

    engine.tokenizer = _FakeTokenizer()
    return engine


def _make_processor():
    return BasicLLMProcessor.__new__(BasicLLMProcessor)


def _run_one_step(engine, processor):
    """跑一个调度 step：真实 schedule + 真实输入构建 + 真实 _update_requests。"""
    out = engine.scheduler.schedule()
    if out is None or not out.scheduled_requests:
        return None, None
    model_input = processor._build_model_input_from_batch_scheduler_output(
        out, 1.0, 0.8, 1
    )
    # 模拟 C++：每个请求在其最后输入位置采一个 token（1:1 对齐）
    sampled = [_SAMPLED_TOKEN] * out.num_requests
    engine._update_requests(
        out.scheduled_requests,
        sampled,
        getattr(out, "num_scheduled_tokens", None),
    )
    return out, model_input


def _run_until_finished(engine, processor, requests, max_steps=50):
    records = []
    for _ in range(max_steps):
        out, model_input = _run_one_step(engine, processor)
        if out is None:
            break
        records.append((out, model_input))
        if all(r.is_finished() for r in requests):
            break
    return records


def _make_scheduler(enable_chunked_prefill, enable_prefix_caching=False, budget=6):
    return Scheduler(
        max_batch_size=8,
        num_blocks=64,
        block_size=4,
        max_num_batched_tokens=budget,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
    )


def test_chunk_split_and_mixed_batch():
    sched = _make_scheduler(enable_chunked_prefill=True)
    engine = _make_engine(sched)
    processor = _make_processor()

    req_a = _make_request("a", range(10), max_tokens=3)
    req_b = _make_request("b", range(1000, 1005), max_tokens=3)
    sched.add_request(req_a)
    sched.add_request(req_b)

    # step 1: 预算 6，A 切 6 个 token 的中段 chunk；B 无预算留在 waiting
    out, mi = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens == [6], out.num_scheduled_tokens
    assert out.is_prefill
    assert mi["input_ids"].data == [list(range(0, 6))]
    assert mi["position_ids"].data == [0, 1, 2, 3, 4, 5]
    assert mi["past_kv_lengths"].data == [0]
    assert mi["total_kv_lengths"].data == [6]
    assert mi["cu_seqlens"].data == [0, 6]
    assert mi["slot_mapping"].data == [0, 1, 2, 3, 4, 5]
    # 中段 chunk：不 append 废 token，只推进 num_computed_tokens
    assert req_a.num_computed_tokens == 6
    assert req_a.get_num_generated_tokens() == 0
    assert req_a.generated_text == ""

    # step 2: B 整段 5 个 token 完成 prefill，剩余预算 1 给 A 续 1 个 token
    out, mi = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens == [5, 1], out.num_scheduled_tokens
    assert out.is_prefill
    assert mi["input_ids"].data == [[1000, 1001, 1002, 1003, 1004, 6]]
    assert mi["past_kv_lengths"].data == [0, 6]
    assert mi["total_kv_lengths"].data == [5, 7]
    assert mi["cu_seqlens"].data == [0, 5, 12]
    assert mi["input_offsets"].data == [0, 5, 6]
    assert req_b.get_num_generated_tokens() == 1  # B 最后一块 chunk，产出首 token
    assert req_a.num_computed_tokens == 7
    assert req_a.get_num_generated_tokens() == 0  # A 仍是中段 chunk

    # step 3: B decode 1 个 token + A 续 3 个 token 的 chunk，混入同一批次
    out, mi = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens == [1, 3], out.num_scheduled_tokens
    assert out.is_prefill  # 派生语义：批次中包含 prefill chunk
    assert mi["input_ids"].data == [[_SAMPLED_TOKEN, 7, 8, 9]]
    assert mi["position_ids"].data == [5, 7, 8, 9]
    assert mi["past_kv_lengths"].data == [5, 7]
    # total_kv_lengths 是本 chunk 结束后的可见 KV 长度，不是整段 prompt 长
    assert mi["total_kv_lengths"].data == [6, 10]
    assert mi["cu_seqlens"].data == [0, 6, 16]
    assert mi["input_offsets"].data == [0, 1, 4]
    assert mi["slot_mapping"].data == [17, 7, 8, 9]
    assert mi["block_tables"].data == [[3, 4, -1], [0, 1, 2]]
    assert req_a.get_num_generated_tokens() == 1  # A 完成 prefill，产出首 token

    # step 4: 纯 decode 批次
    out, mi = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens == [1, 1], out.num_scheduled_tokens
    assert not out.is_prefill

    # 后续步骤预算都不超限，直至两个请求全部完成
    while not (req_a.is_finished() and req_b.is_finished()):
        out, _ = _run_one_step(engine, processor)
        assert out is not None
        assert sum(out.num_scheduled_tokens) <= 6
    assert list(req_a.generated_token_ids) == [_SAMPLED_TOKEN] * 3
    assert list(req_b.generated_token_ids) == [_SAMPLED_TOKEN] * 3
    # 全部结束后可用块数恢复（无泄漏）
    assert sched.cache_manager.get_total_usable_blocks() == 64


def test_legacy_mode_unchanged():
    sched = _make_scheduler(enable_chunked_prefill=False)
    engine = _make_engine(sched)
    processor = _make_processor()

    req_a = _make_request("a", range(10), max_tokens=3)
    sched.add_request(req_a)

    # step 1: 旧逻辑整段 prefill，num_scheduled_tokens 为 None
    out, mi = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens is None
    assert out.is_prefill
    assert mi["input_ids"].data == [list(range(10))]
    assert mi["total_kv_lengths"].data == [10]
    assert req_a.get_num_generated_tokens() == 1

    # step 2: A 在 decode 时来了 B，旧逻辑 prefill/decode 互斥：
    # 本步只排 B 的整段 prefill，A 的 decode 停摆
    req_b = _make_request("b", range(1000, 1005), max_tokens=2)
    sched.add_request(req_b)
    out, mi = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens is None
    assert out.is_prefill
    assert [r.request_id for r in out.scheduled_requests] == ["b"]
    assert mi["input_ids"].data == [list(range(1000, 1005))]
    assert mi["total_kv_lengths"].data == [5]

    # step 3: 纯 decode 批次
    out, mi = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens is None
    assert not out.is_prefill
    assert {r.request_id for r in out.scheduled_requests} == {"a", "b"}

    _run_until_finished(engine, processor, [req_a, req_b])
    assert list(req_a.generated_token_ids) == [_SAMPLED_TOKEN] * 3
    assert list(req_b.generated_token_ids) == [_SAMPLED_TOKEN] * 2
    assert sched.cache_manager.get_total_usable_blocks() == 64


def test_multimodal_request_not_chunked():
    sched = _make_scheduler(enable_chunked_prefill=True)
    engine = _make_engine(sched)
    processor = _make_processor()

    req_a = _make_request("a", range(4), max_tokens=3)
    sched.add_request(req_a)
    # step 1: A 整段 prefill（4 <= 预算 6）
    out, _ = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens == [4]

    req_mm = _make_request(
        "mm", range(1000, 1010), max_tokens=2, has_multimodal_inputs=True
    )
    sched.add_request(req_mm)

    # step 2: A decode 占 1 预算，多模态请求不切分且整体超预算 -> 整请求推迟
    out, mi = _run_one_step(engine, processor)
    assert [r.request_id for r in out.scheduled_requests] == ["a"]
    assert out.num_scheduled_tokens == [1]
    assert not out.is_prefill

    # step 3: 同上，A 生成最后一个 token 后结束
    out, _ = _run_one_step(engine, processor)
    assert [r.request_id for r in out.scheduled_requests] == ["a"]
    assert req_a.is_finished()

    # step 4: running 已空，多模态请求作为单请求保底放行，整段 prefill
    out, mi = _run_one_step(engine, processor)
    assert [r.request_id for r in out.scheduled_requests] == ["mm"]
    assert out.num_scheduled_tokens == [10], out.num_scheduled_tokens
    assert out.is_prefill
    assert mi["total_kv_lengths"].data == [10]
    assert req_mm.get_num_generated_tokens() == 1

    _run_until_finished(engine, processor, [req_mm])
    assert list(req_mm.generated_token_ids) == [_SAMPLED_TOKEN] * 2
    assert sched.cache_manager.get_total_usable_blocks() == 64


def test_prefix_cache_across_chunks():
    sched = _make_scheduler(
        enable_chunked_prefill=True, enable_prefix_caching=True, budget=6
    )
    engine = _make_engine(sched)
    processor = _make_processor()
    prompt_ids = list(range(12))  # block_size=4，共 3 个完整块

    req_a = _make_request("a", prompt_ids, max_tokens=1)
    sched.add_request(req_a)

    # step 1: chunk [0,6)，commit 后只发布第 0 个完整块
    out, _ = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens == [6]
    assert req_a.num_computed_tokens == 6
    assert req_a.num_cache_indexed_blocks == 1
    assert len(sched.cache_manager.hash_to_block_ids) == 1

    # step 2: chunk [6,12)，prefill 完成，3 个块全部发布
    out, _ = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens == [6]
    assert req_a.is_finished()
    assert req_a.num_cache_indexed_blocks == 3
    assert len(sched.cache_manager.hash_to_block_ids) == 3

    # step 3: 相同 prompt 的 B 命中前缀缓存（最多复用到 prompt_len-1），
    # 只需计算 [8,12) 一个 chunk
    req_b = _make_request("b", prompt_ids, max_tokens=1)
    sched.add_request(req_b)
    out, mi = _run_one_step(engine, processor)
    assert out.num_scheduled_tokens == [4]
    assert req_b.num_local_cached_tokens == 8
    assert mi["input_ids"].data == [[8, 9, 10, 11]]
    assert mi["past_kv_lengths"].data == [8]
    assert mi["total_kv_lengths"].data == [12]
    assert req_b.is_finished()
    assert sched.cache_manager.get_total_usable_blocks() == 64


if __name__ == "__main__":
    test_chunk_split_and_mixed_batch()
    print("PASS test_chunk_split_and_mixed_batch")
    test_legacy_mode_unchanged()
    print("PASS test_legacy_mode_unchanged")
    test_multimodal_request_not_chunked()
    print("PASS test_multimodal_request_not_chunked")
    test_prefix_cache_across_chunks()
    print("PASS test_prefix_cache_across_chunks")
    print("ALL OK")
