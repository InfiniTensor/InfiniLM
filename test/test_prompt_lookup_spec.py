"""prompt-lookup 投机采样 + spec×chunked 混排的离线冒烟测试（无需 GPU）。

与 test_chunked_prefill.py 相同的 stub 思路：预置轻量 sys.modules stub，
真实加载 scheduler / cache_manager / request / basic_llm_processor / llm /
speculative_runner。target 模型用确定性伪模型（next token 只由当前 token
决定的 succ 函数，可配 trap）替代，从而可以精确断言：

1. prompt-lookup draft 辅助函数的 n-gram 后缀匹配语义（最近一次出现、
   min/max ngram 边界、k 截断、未命中返回 []）；
2. 投机路径输出与非投机贪心路径逐 token 相同（数学无损），命中时接受率
   计数增长、平均每步产出 > 1；
3. draft 被拒绝（整体/部分）时输出仍然精确，verify 临时槽位被回滚，
   结束后 KV 块无泄漏；
4. spec × chunked prefill 混排：中段 chunk 请求不参与 draft/verify、
   产出 []，decode 请求在同一批次里正常投机，完成 prefill 的请求在
   混排批次里也能投机；
5. 非 greedy 配置回退到常规前向（输出形状为每请求单个 token）；
6. 自适应收益门控：零命中负载攒满窗口后回退常规前向（冷静期内无
   融合前向调用），输出仍与非投机真值逐 token 相同。

直接运行: python3 test/test_prompt_lookup_spec.py
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
        ("infinilm.cache", "cache"),
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

    # 5) infinicore：from_list 只做数据捕获
    infinicore = types.ModuleType("infinicore")
    infinicore.int64 = "int64"
    infinicore.int32 = "int32"

    class _FakeTensor:
        def __init__(self, data, dtype=None):
            self.data = data
            self.dtype = dtype

        def to_numpy(self):
            return self

        def tolist(self):
            return self.data

    infinicore.from_list = lambda data, dtype=None: _FakeTensor(data, dtype)
    infinicore._FakeTensor = _FakeTensor
    sys.modules["infinicore"] = infinicore

    # 6) speculative_runner / llm 的重型依赖（prompt_lookup 路径不会真正调用）
    engine_config = types.ModuleType("infinilm.config.engine_config")
    engine_config.EngineConfig = object
    sys.modules["infinilm.config.engine_config"] = engine_config

    kv_transfer = types.ModuleType("infinilm.config.kv_transfer")
    kv_transfer.KVTransferConfig = object
    sys.modules["infinilm.config.kv_transfer"] = kv_transfer

    infer_engine = types.ModuleType("infinilm.infer_engine")
    infer_engine.model_uses_mamba_cache = lambda hf_config: False
    infer_engine.read_hf_config = lambda model_path: {}
    infer_engine.InferEngine = object
    sys.modules["infinilm.infer_engine"] = infer_engine

    cache_mod = types.ModuleType("infinilm.cache.cache")
    cache_mod.StaticKVCacheConfig = object
    sys.modules["infinilm.cache.cache"] = cache_mod

    distributed = types.ModuleType("infinilm.distributed")
    distributed.DistConfig = object
    sys.modules["infinilm.distributed"] = distributed

    modeling_utils = types.ModuleType("infinilm.modeling_utils")
    modeling_utils.load_model_state_dict_by_file = lambda *a, **k: None
    sys.modules["infinilm.modeling_utils"] = modeling_utils

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
from infinilm.llm.model_runner.speculative_runner import (  # noqa: E402
    SpeculativeRunner,
)
from infinilm.llm.request import InferenceRequest  # noqa: E402
from infinilm.llm.sampling_params import SamplingParams  # noqa: E402
from infinilm.llm.scheduler import Scheduler  # noqa: E402
from infinilm.processors.basic_llm_processor import (  # noqa: E402
    BasicLLMProcessor,
)

_FakeTensor = sys.modules["infinicore"]._FakeTensor


class _FakeTargetEngine:
    """伪 target 模型：next token 只由当前输入 token 决定。

    succ(t) = (t+1) % vocab，traps 可覆盖个别 token 的后继（用来制造
    "prompt 撒谎"的 draft 拒绝场景，或让生成绕回 prompt 片段制造命中）。
    forward = 每请求最后输入位置采一个（对应 sample_all_positions=False）；
    forward_raw = 全位置采样（verify 批次逐位置出 succ）。
    """

    def __init__(self, traps=None, vocab=2000):
        self.traps = traps or {}
        self.vocab = vocab
        # 记录每次 forward_raw 的输入 token 数：融合前向 = b×(k+1)，
        # 两前向的 verify = 候选数×k，据此可断言走了哪条路径
        self.raw_calls = []

    def succ(self, token):
        if token in self.traps:
            return self.traps[token]
        return (token + 1) % self.vocab

    def forward(self, input_ids, **kwargs):
        ids = input_ids.data[0]
        offsets = kwargs["input_offsets"].data
        outs = [
            self.succ(ids[offsets[i + 1] - 1]) for i in range(len(offsets) - 1)
        ]
        return _FakeTensor(outs)

    def forward_raw(self, input_ids, **kwargs):
        ids = input_ids.data[0]
        self.raw_calls.append(len(ids))
        return {
            "output_ids": _FakeTensor([self.succ(t) for t in ids]),
            "logits": None,
            "hidden_states": None,
        }


def _make_runner(fake_engine, num_draft_tokens=3, top_k=1, temperature=1.0):
    config = types.SimpleNamespace(
        speculative_method="prompt_lookup",
        num_draft_tokens=num_draft_tokens,
        max_batch_size=8,
        top_k=top_k,
        temperature=temperature,
    )
    return SpeculativeRunner(config, fake_engine, device=None)


def _make_request(req_id, prompt_token_ids, max_tokens):
    return InferenceRequest(
        request_id=req_id,
        prompt="x" * len(prompt_token_ids),
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        eos_token_ids=[],
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


def _make_scheduler(enable_chunked_prefill, budget=6):
    return Scheduler(
        max_batch_size=8,
        num_blocks=64,
        block_size=4,
        max_num_batched_tokens=budget,
        enable_prefix_caching=False,
        enable_chunked_prefill=enable_chunked_prefill,
    )


def _run_one_step(engine, processor, runner):
    """跑一个调度 step：真实 schedule + 真实输入构建 + 真实 runner + 真实
    _update_requests。返回 (scheduler_output, runner 输出)。"""
    out = engine.scheduler.schedule()
    if out is None or not out.scheduled_requests:
        return None, None
    model_input = processor._build_model_input_from_batch_scheduler_output(
        out, 1.0, 0.8, 1
    )
    sampled = runner.forward(out, model_input)
    engine._update_requests(
        out.scheduled_requests,
        sampled,
        getattr(out, "num_scheduled_tokens", None),
    )
    return out, sampled


def _run_until_finished(engine, processor, runner, requests, max_steps=500):
    for _ in range(max_steps):
        out, _ = _run_one_step(engine, processor, runner)
        if out is None:
            break
        if all(r.is_finished() for r in requests):
            break


def _truth(fake, last_prompt_token, n):
    """非投机贪心路径的真值：从 prompt 尾 token 起逐次 succ。"""
    out = []
    t = last_prompt_token
    for _ in range(n):
        t = fake.succ(t)
        out.append(t)
    return out


def test_lookup_helper():
    runner = _make_runner(_FakeTargetEngine())
    # 默认 env：max_ngram=4, min_ngram=2
    req = _make_request("u", [10, 11, 12, 13, 10, 11, 12, 55], max_tokens=8)

    # 序列 [10,11,12,13,10,11,12,55]，当前后缀指向末尾
    req._generated_token_ids.extend([13, 10, 11])
    # context = [10,11,12,13,10,11,12,55,13,10,11]，后缀 [13,10,11]：
    # i=3 处 context[3:6]=[13,10,11] 命中（最近一次），其后是 [12,55,...]
    drafts = runner._draft_prompt_lookup_tokens(req, 2)
    assert drafts == [12, 55], drafts

    # k 截断：只取 1 个
    drafts = runner._draft_prompt_lookup_tokens(req, 1)
    assert drafts == [12], drafts

    # 未命中：后缀不存在于前文
    req2 = _make_request("u2", [1, 2, 3, 4], max_tokens=8)
    req2._generated_token_ids.extend([99, 100])
    assert runner._draft_prompt_lookup_tokens(req2, 3) == []

    # min_ngram 以下不匹配（默认 min=2，单 token 后缀不匹配）
    req3 = _make_request("u3", [7, 8, 9], max_tokens=8)
    req3._generated_token_ids.extend([8])
    # context=[7,8,9,8]，后缀 [9,8] 无命中；size=1 不达 min，返回 []
    assert runner._draft_prompt_lookup_tokens(req3, 3) == []

    # 序列太短（不足 min_ngram+1）直接返回 []
    req4 = _make_request("u4", [5], max_tokens=8)
    assert runner._draft_prompt_lookup_tokens(req4, 3) == []


def test_decode_exact_with_acceptance():
    """高命中场景：prompt 内含 wrap 片段，wrap 后 draft 持续全接受。"""
    fake = _FakeTargetEngine(vocab=64)
    sched = _make_scheduler(enable_chunked_prefill=False)
    engine = _make_engine(sched)
    processor = BasicLLMProcessor.__new__(BasicLLMProcessor)
    runner = _make_runner(fake, num_draft_tokens=3)

    prompt = [60, 61, 62, 63, 0, 1, 2, 3]
    req = _make_request("a", prompt, max_tokens=130)
    sched.add_request(req)
    _run_until_finished(engine, processor, runner, [req])

    expected = _truth(fake, prompt[-1], 130)
    assert list(req.generated_token_ids) == expected, (
        list(req.generated_token_ids)[:16],
        expected[:16],
    )
    stats = runner.get_acceptance_stats()
    assert stats["drafted_tokens"] > 0
    assert stats["accepted_tokens"] > 0
    # 前 60 步 suffix 是新内容（无命中）；wrap 之后 suffix 都能在
    # prompt/已生成片段里命中，每步产出 k+1=4 个 token，全程平均应显著 > 1
    assert stats["accept_rate"] > 0.9, stats
    assert stats["avg_tokens_per_step"] > 1.4, stats
    # 纯 decode 批次全部走融合单前向（b=1 行 × (k+1)=4 token），
    # 不应出现两前向流程的 verify 调用（1×k=3 token）
    assert fake.raw_calls, "decode 批次应产生融合前向调用"
    assert all(n == 4 for n in fake.raw_calls), fake.raw_calls
    assert sched.cache_manager.get_total_usable_blocks() == 64


def test_rejection_and_rollback():
    """拒绝路径：draft 整体被拒绝 / 部分接受，输出仍与真值逐 token 相同。"""
    # trap 让生成绕回 prompt 中的片段；prompt 在关键位置"撒谎"
    fake = _FakeTargetEngine(traps={999: 20, 888: 30}, vocab=2000)
    sched = _make_scheduler(enable_chunked_prefill=False)
    engine = _make_engine(sched)
    processor = BasicLLMProcessor.__new__(BasicLLMProcessor)
    runner = _make_runner(fake, num_draft_tokens=2)

    # r1: prompt=[20,21,22,77,999]。生成 20,21 后 suffix [20,21] 命中，
    # draft=[22,77]；d1=22 接受，d2=77 被拒绝（真值 23），修正为 23。
    r1 = _make_request("r1", [20, 21, 22, 77, 999], max_tokens=7)
    # r2: prompt=[30,31,55,888]。生成 30,31 后 suffix [30,31] 命中，
    # draft=[55,888]；d1=55 与 target 32 不符，整体拒绝，只产 32。
    r2 = _make_request("r2", [30, 31, 55, 888], max_tokens=5)
    sched.add_request(r1)
    sched.add_request(r2)
    _run_until_finished(engine, processor, runner, [r1, r2])

    assert list(r1.generated_token_ids) == [20, 21, 22, 23, 24, 25, 26]
    assert list(r2.generated_token_ids) == [30, 31, 32, 33, 34]
    stats = runner.get_acceptance_stats()
    assert stats["drafted_tokens"] >= 4, stats  # 两条 draft 链都被验证过
    assert stats["accepted_tokens"] >= 1, stats  # r1 的 d1 被接受
    assert stats["avg_tokens_per_step"] > 1.0, stats
    # verify 槽位回滚 + 请求结束释放后无块泄漏
    assert sched.cache_manager.get_total_usable_blocks() == 64


def test_spec_with_chunked_mixed_batch():
    """spec × chunked prefill 混排：中段 chunk 不投机、不产 token，
    decode 请求同批次正常投机，完成 prefill 的请求也能立刻投机。"""
    fake = _FakeTargetEngine(traps={9: 5}, vocab=2000)
    sched = _make_scheduler(enable_chunked_prefill=True, budget=6)
    engine = _make_engine(sched)
    processor = BasicLLMProcessor.__new__(BasicLLMProcessor)
    runner = _make_runner(fake, num_draft_tokens=3)

    # A：长 prompt 被切块（10 > 预算 6），prompt 内含自重复片段
    req_a = _make_request("a", [100, 101, 102, 103, 100, 101, 102, 103, 100, 101], max_tokens=6)
    # B：trap succ(9)=5 形成周期 5 循环，decode 期持续命中
    req_b = _make_request("b", [5, 6, 7, 8, 9, 5, 6, 7, 8, 9], max_tokens=12)
    sched.add_request(req_a)
    sched.add_request(req_b)

    # step 1: A 切 6 token 中段 chunk，B 无预算留在 waiting
    out, sampled = _run_one_step(engine, processor, runner)
    assert out.num_scheduled_tokens == [6]
    assert sampled == [[]], sampled  # 中段 chunk：runner 产出空，不投机
    assert req_a.get_num_generated_tokens() == 0
    assert req_a.num_computed_tokens == 6

    # step 2: A 被重新入队排在 B 后面，B 先切 6 token 中段 chunk
    out, sampled = _run_one_step(engine, processor, runner)
    assert out.num_scheduled_tokens == [6], out.num_scheduled_tokens
    assert sampled == [[]], sampled
    assert req_b.get_num_generated_tokens() == 0
    assert req_b.num_computed_tokens == 6

    # step 3: A 续 4 token 完成 prefill 并投机产出，B 只分到 2 token
    # （预算被 A 占去 4），仍是中段 chunk
    out, sampled = _run_one_step(engine, processor, runner)
    assert out.num_scheduled_tokens == [4, 2], out.num_scheduled_tokens
    assert sampled[0] == [102, 103, 104], sampled  # A 完成块即投机（d3 被拒）
    assert sampled[1] == [], sampled
    assert req_b.get_num_generated_tokens() == 0

    # step 4: A decode 占 1 预算（正常投机），B 续 2 token 完成 prefill 并投机
    out, sampled = _run_one_step(engine, processor, runner)
    assert out.num_scheduled_tokens == [1, 2], out.num_scheduled_tokens
    assert sampled[0] == [105], sampled
    assert sampled[1] == [5, 6, 7, 8], sampled  # B 完成块即投机（全接受）

    # 后续步骤自由推进（纯 decode 混排）直至全部完成
    _run_until_finished(engine, processor, runner, [req_a, req_b])

    assert list(req_a.generated_token_ids) == [102, 103, 104, 105, 106, 107]
    assert list(req_b.generated_token_ids) == [5, 6, 7, 8, 9, 5, 6, 7, 8, 9, 5, 6]
    stats = runner.get_acceptance_stats()
    assert stats["accepted_tokens"] > 0, stats
    assert stats["avg_tokens_per_step"] > 1.2, stats
    # 混排批次走两前向（verify：候选×k token），纯 decode 批次走融合
    # （b×(k+1)）：两种形状的 forward_raw 调用都应出现
    assert any(n in (3, 6) for n in fake.raw_calls), fake.raw_calls
    assert any(n in (4, 8) for n in fake.raw_calls), fake.raw_calls
    assert sched.cache_manager.get_total_usable_blocks() == 64


def test_legacy_prefill_batch_spec():
    """旧调度（非 chunked）的整段 prefill 批次同样能投机。"""
    fake = _FakeTargetEngine(traps={9: 5}, vocab=2000)
    sched = _make_scheduler(enable_chunked_prefill=False)
    engine = _make_engine(sched)
    processor = BasicLLMProcessor.__new__(BasicLLMProcessor)
    runner = _make_runner(fake, num_draft_tokens=3)

    req = _make_request("a", [5, 6, 7, 8, 9, 5, 6, 7, 8, 9], max_tokens=8)
    sched.add_request(req)

    # step 1: 整段 prefill，prompt 内自重复使 draft 全接受
    out, sampled = _run_one_step(engine, processor, runner)
    assert out.is_prefill
    assert sampled == [[5, 6, 7, 8]], sampled

    _run_until_finished(engine, processor, runner, [req])
    assert list(req.generated_token_ids) == [5, 6, 7, 8, 9, 5, 6, 7]
    assert sched.cache_manager.get_total_usable_blocks() == 64


def test_nongreedy_fallback():
    """非 greedy 配置：回退到常规前向，每请求产出单个 token（平铺列表）。"""
    fake = _FakeTargetEngine(traps={9: 5}, vocab=2000)
    sched = _make_scheduler(enable_chunked_prefill=False)
    engine = _make_engine(sched)
    processor = BasicLLMProcessor.__new__(BasicLLMProcessor)
    runner = _make_runner(fake, num_draft_tokens=3, temperature=0.7)

    req = _make_request("a", [5, 6, 7, 8, 9], max_tokens=4)
    sched.add_request(req)
    _run_until_finished(engine, processor, runner, [req])

    assert list(req.generated_token_ids) == [5, 6, 7, 8]
    stats = runner.get_acceptance_stats()
    assert stats["drafted_tokens"] == 0  # 从未进入投机路径
    assert sched.cache_manager.get_total_usable_blocks() == 64


def test_adaptive_gate():
    """低收益场景自适应回退：无 n-gram 命中时平均每步只产 1 token，
    攒满窗口（默认 32 请求步）后触发门控，冷静期（默认 64 步）内走
    常规前向、不再有融合 verify 调用，输出仍与真值逐 token 相同。"""
    fake = _FakeTargetEngine(vocab=2000)
    sched = _make_scheduler(enable_chunked_prefill=False)
    engine = _make_engine(sched)
    processor = BasicLLMProcessor.__new__(BasicLLMProcessor)
    runner = _make_runner(fake, num_draft_tokens=3)

    # 严格递增序列（vocab 足够大不 wrap）：任何后缀在前文中都不重复，
    # draft 永不命中 → 每步只产 1 token
    req = _make_request("g", [1000, 1001], max_tokens=80)
    sched.add_request(req)
    _run_until_finished(engine, processor, runner, [req])

    assert list(req.generated_token_ids) == _truth(fake, 1001, 80)
    stats = runner.get_acceptance_stats()
    assert stats["gate_triggered"] == 1, stats
    # 第 32 步触发门控，之后 48 个 token 全部走常规前向。融合前向
    # （forward_raw）停留在 31 次：第 1 步是整段 prefill 批次，走两前向
    # 流程且无 draft 命中、不产生 verify 调用；第 2..32 步才是融合前向
    assert len(fake.raw_calls) == 31, len(fake.raw_calls)
    assert sched.cache_manager.get_total_usable_blocks() == 64


if __name__ == "__main__":
    test_lookup_helper()
    print("PASS test_lookup_helper")
    test_decode_exact_with_acceptance()
    print("PASS test_decode_exact_with_acceptance")
    test_rejection_and_rollback()
    print("PASS test_rejection_and_rollback")
    test_spec_with_chunked_mixed_batch()
    print("PASS test_spec_with_chunked_mixed_batch")
    test_legacy_prefill_batch_spec()
    print("PASS test_legacy_prefill_batch_spec")
    test_nongreedy_fallback()
    print("PASS test_nongreedy_fallback")
    test_adaptive_gate()
    print("PASS test_adaptive_gate")
    print("ALL OK")
