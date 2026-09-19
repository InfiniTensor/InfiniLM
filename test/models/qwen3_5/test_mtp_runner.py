"""CPU control-flow checks using real scheduling, pages and request updates.

The deterministic engine below checks orchestration, not GPU numerical accuracy.
"""

from types import SimpleNamespace

import infinicore
import pytest
from infinilm.config.engine_config import EngineConfig
from infinilm.llm.llm import LLMEngine
from infinilm.llm.model_runner.model_runner import ModelRunner
from infinilm.llm.model_runner.mtp_runner import MTPRunner
from infinilm.llm.request import InferenceRequest, RequestStatus
from infinilm.llm.sampling_params import SamplingParams
from infinilm.llm.scheduler import Scheduler
from infinilm.processors.qwen3_5_processor import Qwen35Processor


def advance(state, token):
    return (3 * state + token) % 97


def serial_tokens(prompt, count):
    state = 0
    for token in prompt:
        state = advance(state, token)
    output = []
    for _ in range(count):
        token = (state + 13) % 97
        output.append(token)
        state = advance(state, token)
    return output


class DeterministicEngine:
    position_id_axes = 3

    def __init__(self, policy):
        self.rows = {0: 0}
        self.policy = policy
        self.draft_calls = 0

    def forward_raw(self, **inputs):
        ids = inputs["input_ids"].to_numpy()[0].tolist()
        hidden = inputs.get("target_hidden_states")
        states = []
        if hidden is None:
            offsets = inputs["input_offsets"].to_numpy().tolist()
            initial = inputs["mamba_init_state_indices"].to_numpy().tolist()
            finals = inputs["mamba_final_state_indices"].to_numpy().tolist()
            destinations = inputs.get("token_state_indices")
            destinations = (
                destinations.to_numpy().tolist() if destinations is not None else None
            )
            sampled = []
            for r, (begin, end) in enumerate(zip(offsets, offsets[1:])):
                state = self.rows[initial[r]]
                for i in range(begin, end):
                    state = advance(state, ids[i])
                    states.append(state)
                    if destinations is not None:
                        self.rows[destinations[i]] = state
                self.rows[finals[r]] = state
            if inputs.get("sample_all_positions", False):
                sampled = [(state + 13) % 97 for state in states]
            else:
                sampled = [(states[end - 1] + 13) % 97 for end in offsets[1:]]
        else:
            self.draft_calls += 1
            previous = hidden.to_numpy()[0, :, 0].tolist()
            states = [advance(int(state), token) for state, token in zip(previous, ids)]
            reject = (
                (isinstance(self.policy, int) and self.draft_calls == self.policy + 1)
                or self.policy == "reject"
                or (self.policy == "alternate" and self.draft_calls % 2 == 0)
            )
            sampled = [(state + 13 + int(reject)) % 97 for state in states]
        if hidden is not None and not inputs.get("sample_all_positions", False):
            offsets = inputs["input_offsets"].to_numpy().tolist()
            sampled = [sampled[end - 1] for end in offsets[1:]]
        return {
            "output_ids": infinicore.from_list(sampled, dtype=infinicore.int64),
            "hidden_states": infinicore.from_list(
                [[state] for state in states], dtype=infinicore.float32
            ).view((1, len(states), 1)),
        }


def service(policy="alternate", candidates=1, *, batch_size=1, state_rows=0):
    config = EngineConfig(
        "unused",
        enable_mtp=True,
        num_draft_tokens=candidates,
        max_batch_size=batch_size,
        num_state_rows=state_rows,
        enable_prefix_caching=False,
        num_blocks=max(16, 4 * (candidates + 3)),
        block_size=4,
    )
    engine = DeterministicEngine(policy)
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner.model_engine = engine
    runner.speculative_runner = MTPRunner(config, engine)
    runner.processor = Qwen35Processor.__new__(Qwen35Processor)
    runner.kv_connector = None
    runner.pipeline_control = None
    result = LLMEngine.__new__(LLMEngine)
    result.config = config
    result.model_runner = runner
    result.scheduler = Scheduler(
        max_batch_size=config.max_batch_size,
        num_blocks=config.num_blocks,
        block_size=config.block_size,
        num_mamba_cache_blocks=config.num_state_rows,
        has_mamba_cache=True,
        enable_prefix_caching=False,
    )
    result.tokenizer = SimpleNamespace(
        decode=lambda ids: "".join(chr(0x4E00 + token) for token in ids)
    )
    result.eos_token_ids = []
    return result


def request(name="a", prompt=None, **sampling):
    return InferenceRequest(
        name,
        prompt_token_ids=prompt or [3, 8, 15],
        sampling_params=SamplingParams(max_tokens=9, **sampling),
    )


def drain(engine, req):
    while not req.is_finished():
        assert engine.step()[0]
    return list(req.generated_token_ids)


def assert_released(engine):
    assert not engine.scheduler.mamba_cache_manager.used_block_ids
    manager = engine.scheduler.cache_manager
    assert all(block.ref_count == 0 for block in manager.blocks)
    assert manager.get_total_usable_blocks() == manager.num_blocks


@pytest.mark.parametrize("candidates,policy", [(2, "reject"), (4, "alternate")])
def test_serial_equivalence_across_page_boundaries(candidates, policy):
    engine, req = service(policy, candidates), request()
    engine.add_request(req)
    while not req.is_finished():
        engine.step()
        if not req.is_finished():
            assert req.mtp_state.cached_tokens == req.get_total_length() - 1
    assert list(req.generated_token_ids) == serial_tokens(req.prompt_token_ids, 9)
    assert_released(engine)


@pytest.mark.parametrize("stop_kind", ["eos", "string", "length", "prefill"])
def test_stopping_truncates_output_and_releases_state(stop_kind):
    engine, req = service("accept", candidates=4), request()
    expected = serial_tokens(req.prompt_token_ids, 9)
    limit = 1 if stop_kind == "prefill" else 2
    if stop_kind == "eos":
        req.eos_token_ids = [expected[1]]
    elif stop_kind == "string":
        req.sampling_params.stop = [chr(0x4E00 + expected[1])]
    else:
        req.sampling_params.max_tokens = limit
    output_queue = req.output_queue
    engine.add_request(req)
    outputs = []
    while not req.is_finished():
        _, pending = engine.step()
        outputs.extend(output for _, output in pending)
    assert list(req.generated_token_ids) == expected[:limit]
    assert [output.token_id for output in outputs] == expected[:limit]
    assert [output.finished for output in outputs] == [False] * (limit - 1) + [True]
    assert_released(engine)
    output_queue.close()


def test_kv_capacity_falls_back_then_resumes_mtp():
    engine, req = service("accept", candidates=2), request()
    engine.add_request(req)
    engine.step()
    manager = engine.scheduler.cache_manager
    occupied, _ = manager.allocate_slots(manager.get_num_free_blocks() * 4)
    engine.step()  # pending is in the last slot; speculative tail needs a page
    assert engine.model_runner.speculative_runner.num_capacity_fallbacks == 1
    manager.free_blocks(occupied)
    assert drain(engine, req) == serial_tokens(req.prompt_token_ids, 9)
    assert engine.model_runner.speculative_runner.num_proposals > 0
    assert_released(engine)


def test_state_capacity_uses_normal_decode_without_stale_draft_cache():
    engine, req = service(candidates=2), request()
    manager = engine.scheduler.mamba_cache_manager
    occupied = [manager.allocate(), manager.allocate()]
    engine.add_request(req)
    assert drain(engine, req) == serial_tokens(req.prompt_token_ids, 9)
    assert engine.model_runner.model_engine.draft_calls == 0
    for row in occupied:
        manager.free(row)
    assert_released(engine)


@pytest.mark.parametrize(
    "option",
    [
        {"enable_prefix_caching": True},
        {"num_draft_tokens": 5},
        {"num_draft_tokens": 2, "enable_graph": True},
        {"max_batch_size": 2, "enable_graph": True},
        {"pipeline_parallel_size": 2},
        {"cache_type": "static"},
        {"draft_model_path": "external"},
        {"top_k": 8},
        {"attn_backend": "flash-attn"},
    ],
)
def test_unsupported_engine_modes_fail_before_loading(option):
    options = dict(
        enable_mtp=True,
        num_draft_tokens=1,
        max_batch_size=1,
        enable_prefix_caching=False,
    )
    options.update(option)
    with pytest.raises(ValueError):
        EngineConfig("unused", **options)


@pytest.mark.parametrize(
    "candidates,prefix", [(k, n) for k in (1, 2, 4) for n in range(k + 1)]
)
def test_every_acceptance_prefix_selects_checkpoint_without_target_replay(
    candidates, prefix
):
    engine, req = service(prefix, candidates), request()
    engine.add_request(req)
    engine.step()
    rows = list(req.mtp_state.scratch_indices)
    runner = engine.model_runner.speculative_runner
    engine.step()
    assert runner.num_accepted == prefix
    assert runner.num_proposals == candidates
    assert runner.num_target_calls == 2
    assert req.mamba_cache_index == rows[prefix]
    assert drain(engine, req) == serial_tokens(req.prompt_token_ids, 9)
    assert_released(engine)


@pytest.mark.parametrize("small_pool", [False, True])
def test_packed_requests_arrival_cancel_and_capacity_fallback(small_pool):
    engine = service(
        "alternate", candidates=2, batch_size=3, state_rows=5 if small_pool else 0
    )
    requests = [
        request("a"),
        request("b", prompt=[7, 2, 10]),
        request("c", prompt=[19, 9, 13]),
    ]
    engine.add_request(requests[0])
    engine.add_request(requests[1])
    engine.step()
    engine.step()
    requests[0].mark_canceled()
    engine.add_request(requests[2])
    for _ in range(40):
        if all(r.is_finished() for r in requests):
            break
        assert engine.step()[0]
    for req in requests[1:]:
        assert list(req.generated_token_ids) == serial_tokens(req.prompt_token_ids, 9)
    assert_released(engine)
    if small_pool:
        assert engine.model_runner.speculative_runner.num_capacity_fallbacks > 0


def test_shutdown_reclaims_running_and_waiting_requests_once():
    engine = service(candidates=4)
    engine.model_runner._closed = False
    active, waiting = request("active"), request("waiting")
    engine.add_request(active)
    engine.step()
    engine.step()  # rotate checkpoint ownership
    engine.add_request(waiting)
    engine.close()
    engine.close()
    assert active.status == waiting.status == RequestStatus.CANCELED
    assert engine.scheduler.waiting_queue.sync_q.empty()
    assert engine.scheduler.running_queue.sync_q.empty()
    assert_released(engine)
    # Repeat completion after those pages have a different owner.
    blocks, _ = engine.scheduler.cache_manager.allocate_slots(4)
    engine.scheduler.complete_requests([active])
    assert all(engine.scheduler.cache_manager.blocks[b].ref_count == 1 for b in blocks)
    engine.scheduler.cache_manager.free_blocks(blocks)
    with pytest.raises(RuntimeError, match="closed"):
        engine.add_request(request("after-close"))


@pytest.mark.parametrize("failure", ["prefill", "verify", "draft"])
def test_packed_failure_releases_all_request_ownership(failure):
    engine = service(candidates=2, batch_size=2)
    requests = [request("a"), request("b")]
    outputs = [req.output_queue for req in requests]
    for req in requests:
        engine.add_request(req)
    if failure != "prefill":
        engine.step()
    forward = engine.model_runner.model_engine.forward_raw

    def fail(**kwargs):
        if failure == "prefill" or (
            failure == "verify" and kwargs.get("sample_all_positions")
        ):
            raise RuntimeError("packed failure")
        if failure == "draft" and kwargs.get("target_hidden_states") is not None:
            raise RuntimeError("packed failure")
        return forward(**kwargs)

    engine.model_runner.model_engine.forward_raw = fail
    with pytest.raises(RuntimeError, match="packed failure"):
        engine.step()
    for req, queue in zip(requests, outputs):
        assert req.status == RequestStatus.FAILED
        output = queue.sync_q.get_nowait()
        assert output.finished and output.token_id == -1
        assert output.finish_reason == req.finish_reason
        req.output_queue.close()
    assert_released(engine)


def test_mtp_config_sizes_state_pool_independently_from_pages():
    config = EngineConfig(
        "unused",
        enable_mtp=True,
        num_draft_tokens=4,
        max_batch_size=3,
        num_blocks=512,
        enable_prefix_caching=False,
    )
    assert config.num_state_rows == 19
    config = EngineConfig(
        "unused",
        enable_mtp=True,
        num_state_rows=5,
        enable_prefix_caching=True,
        mtp_prefix_cache_bytes=1024,
    )
    assert config.num_state_rows == 5
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        EngineConfig(
            "unused",
            enable_mtp=True,
            tensor_parallel_size=2,
            mtp_prefix_cache_bytes=1024,
        )


def test_packed_draft_preserves_different_acceptance_lengths():
    engine = service("accept", candidates=2, batch_size=2)
    raw = engine.model_runner.model_engine
    original = raw.forward_raw
    packed_lengths = []

    def traced(**inputs):
        result = original(**inputs)
        if inputs.get("target_hidden_states") is not None:
            offsets = inputs["input_offsets"].to_numpy().tolist()
            if len(offsets) > 2:
                packed_lengths.append([b - a for a, b in zip(offsets, offsets[1:])])
                # Reject only the first request's rolled candidate. Rebuilding
                # the histories must then pack two different accepted lengths.
                tokens = result["output_ids"].to_numpy().tolist()
                tokens[0] = (tokens[0] + 1) % 97
                result["output_ids"] = infinicore.from_list(
                    tokens, dtype=infinicore.int64
                )
        return result

    raw.forward_raw = traced
    # Caller-supplied IDs need not be unique; ownership follows the request.
    reqs = [request("same"), request("same", prompt=[7, 2, 10])]
    for req in reqs:
        engine.add_request(req)
    while not all(req.is_finished() for req in reqs):
        engine.step()
    assert [1, 1] in packed_lengths
    assert any(len(set(lengths)) > 1 for lengths in packed_lengths)
    for req in reqs:
        assert list(req.generated_token_ids) == serial_tokens(req.prompt_token_ids, 9)
    assert_released(engine)


@pytest.mark.parametrize("model_type,layers", [("qwen3_5_moe", 1), ("qwen3_5", 2)])
def test_unsupported_mtp_architecture_fails_before_worker_setup(
    monkeypatch, model_type, layers
):
    from infinilm.infer_engine import InferEngine

    monkeypatch.setattr(
        "infinilm.infer_engine.read_hf_config",
        lambda _: {
            "model_type": model_type,
            "text_config": {"mtp_num_hidden_layers": layers},
        },
    )
    with pytest.raises(ValueError, match="single-layer dense"):
        InferEngine("unused", enable_mtp=True)


@pytest.mark.parametrize(
    "sampling,error",
    [
        ({"top_k": 8}, "greedy"),
        ({"max_tokens": 0}, "positive integer"),
        ({"max_tokens": -1}, "positive integer"),
        ({"max_tokens": None}, "positive integer"),
        ({"max_tokens": 1.5}, "positive integer"),
        ({"max_tokens": 10000}, "capacity"),
    ],
)
def test_invalid_request_rejected_before_admission(sampling, error):
    engine, req = service(), request()
    for name, value in sampling.items():
        setattr(req.sampling_params, name, value)
    with pytest.raises(ValueError, match=error):
        engine.add_request(req)
    assert engine.scheduler.waiting_queue.sync_q.empty()
    assert_released(engine)
