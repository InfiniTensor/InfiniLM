"""Opt-in native chunk/output/cancellation regression for a dense FP16 model."""

import argparse
from unittest.mock import patch


def check(model, tp=1, chunk_size=17, graph=False):
    import infinicore
    from infinilm.config.engine_config import EngineConfig
    from infinilm.lib import _infinilm
    from infinilm.llm.llm import LLMEngine
    from infinilm.llm.request import InferenceRequest
    from infinilm.llm.sampling_params import SamplingParams

    outputs = []
    for chunk in (0, chunk_size):
        engine = LLMEngine(
            EngineConfig(
                model,
                device="cuda",
                dtype="float16",
                tensor_parallel_size=tp,
                enable_graph=graph,
                attn_backend="paged-attn",
                num_blocks=16,
                block_size=64,
                max_batch_size=1,
                prefill_chunk_size=chunk,
                enable_prefix_caching=True,
                prefix_cache_policy="slru",
            )
        )
        raw = engine.model_runner.model_engine
        native = _infinilm.InferEngine.forward
        calls = []

        def forward(instance, inputs):
            result = native(instance, inputs)
            calls.append(inputs.prefill_only)
            if inputs.prefill_only:
                assert (
                    not result.output_ids
                    and not result.logits
                    and not result.hidden_states
                )
            else:
                assert result.output_ids and result.logits
            return result

        def generate(name, tokens, cancel=False, reused=False):
            request = InferenceRequest(
                name,
                prompt_token_ids=tokens,
                sampling_params=SamplingParams(max_tokens=4, ignore_eos=True, top_k=1),
            )
            engine.add_request(request)
            for step in range(100):
                if request.is_finished():
                    break
                assert engine.step()[0]
                if step == 0 and reused:
                    assert request.num_local_cached_tokens == 64
                if cancel and step == 0:
                    assert not request.generated_token_ids
                    request.abort()
            assert request.is_finished()
            cache = engine.scheduler.cache_manager
            assert all(block.ref_count == 0 for block in cache.blocks)
            assert cache.get_total_usable_blocks() == cache.num_blocks
            if cancel:
                assert (
                    request.status.name == "CANCELED"
                    and not request.generated_token_ids
                )
            else:
                assert len(request.generated_token_ids) == 4
            return list(request.generated_token_ids)

        try:
            caches = _infinilm.InferEngine.get_kv_cache(raw)
            assert len(caches) == tp
            for rank, tensors in enumerate(caches):
                assert {infinicore.Tensor(t).device.index for t in tensors if t} == {
                    rank
                }
            # Invalid output suppression must fail before dispatching worker jobs.
            for arguments, message in (
                (
                    {"prefill_only": True, "sample_all_positions": True},
                    "sample_all_positions=false",
                ),
                ({"prefill_only": True}, "input_offsets"),
            ):
                try:
                    native(raw, _infinilm.InferEngine.Input(**arguments))
                except ValueError as error:
                    assert message in str(error)
                else:
                    raise AssertionError("Invalid outputless forward was accepted.")
            with patch.object(_infinilm.InferEngine, "forward", forward):
                tokens = list(range(1, 68))
                result = generate("first", tokens)
                assert generate("prefix-reuse", tokens, reused=True) == result
                if chunk:
                    assert any(calls) and not all(calls)
                    generate("cancel", [7] * len(tokens), cancel=True)
                else:
                    assert not any(calls)
                outputs.append(result)
        finally:
            engine.close()
    assert outputs[0] == outputs[1], "Chunked and ordinary greedy tokens differ."


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=17)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    if not 0 < args.chunk_size < 67:
        parser.error(
            "--chunk-size must be between 1 and 66 to exercise intermediate chunks"
        )
    check(args.model, args.tp, args.chunk_size, args.graph)
    print("Native chunk output, prefix reuse, cancellation and reclamation passed.")
