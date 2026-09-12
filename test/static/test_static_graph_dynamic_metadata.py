import math
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def function_body(text: str, signature: str) -> str:
    start = text.find(signature)
    if start < 0:
        raise AssertionError(f"missing function: {signature}")
    brace = text.index("{", start)
    depth = 0
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[brace : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def scalar_attention(query: float, keys: list[float], values: list[float]) -> float:
    scores = [query * key for key in keys]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = sum(weights)
    return sum(weight * value for weight, value in zip(weights, values)) / denominator


def device_dynamic_decode(
    key_cache: list[float],
    value_cache: list[float],
    *,
    query: float,
    key: float,
    value: float,
    past_length: int,
    total_length: int,
) -> float:
    key_cache[past_length] = key
    value_cache[past_length] = value
    return scalar_attention(query, key_cache[:total_length], value_cache[:total_length])


def host_captured_decode(
    key_cache: list[float],
    value_cache: list[float],
    *,
    query: float,
    key: float,
    value: float,
    captured_past_length: int,
    captured_total_length: int,
) -> float:
    key_cache[captured_past_length] = key
    value_cache[captured_past_length] = value
    return scalar_attention(
        query,
        key_cache[:captured_total_length],
        value_cache[:captured_total_length],
    )


class StaticGraphDynamicMetadataTest(unittest.TestCase):
    def test_changing_decode_lengths_distinguish_dynamic_from_captured_metadata(self):
        dynamic_k = [0.0, 0.0, 0.0]
        dynamic_v = [0.0, 0.0, 0.0]
        captured_k = [0.0, 0.0, 0.0]
        captured_v = [0.0, 0.0, 0.0]

        first_dynamic = device_dynamic_decode(
            dynamic_k,
            dynamic_v,
            query=1.0,
            key=1.0,
            value=2.0,
            past_length=0,
            total_length=1,
        )
        first_captured = host_captured_decode(
            captured_k,
            captured_v,
            query=1.0,
            key=1.0,
            value=2.0,
            captured_past_length=0,
            captured_total_length=1,
        )
        self.assertAlmostEqual(first_dynamic, first_captured)

        second_dynamic = device_dynamic_decode(
            dynamic_k,
            dynamic_v,
            query=1.0,
            key=3.0,
            value=7.0,
            past_length=1,
            total_length=2,
        )
        second_captured = host_captured_decode(
            captured_k,
            captured_v,
            query=1.0,
            key=3.0,
            value=7.0,
            captured_past_length=0,
            captured_total_length=1,
        )
        self.assertNotAlmostEqual(second_dynamic, second_captured)
        self.assertEqual(dynamic_k[:2], [1.0, 3.0])
        self.assertEqual(captured_k[:2], [3.0, 0.0])

    def test_recording_path_forwards_device_metadata_to_graph_safe_ops(self):
        text = source("csrc/layers/attention/backends/static_attn.cpp")
        forward = function_body(text, "StaticAttentionImpl::forward(")
        recording = forward.find("context::isGraphRecording()")
        host_update = forward.find("do_kv_cache_update(")
        self.assertGreaterEqual(recording, 0)
        self.assertGreater(host_update, recording)
        self.assertIn("return forward_graph_", forward[:host_update])

        graph = function_body(text, "StaticAttentionImpl::forward_graph_(")
        compact = re.sub(r"\s+", "", graph)
        self.assertRegex(
            compact,
            r"op::kv_caching_\([^;]*attn_metadata\.past_sequence_lengths\.value\(\)\);",
        )
        self.assertRegex(
            compact,
            r"op::paged_attention_\([^;]*attn_metadata\.block_tables\.value\(\),"
            r"attn_metadata\.total_sequence_lengths\.value\(\),",
        )
        self.assertNotIn("Device::cpu", graph)
        self.assertNotIn("reinterpret_cast", graph)

    def test_compiler_owns_initialized_i32_replay_metadata(self):
        text = source("csrc/engine/compiler/static_batching_compiler.cpp")
        compile_body = function_body(text, "void StaticBatchingCompiler::compile()")
        replay_body = function_body(text, "StaticBatchingCompiler::get_compiled(")
        compact = re.sub(r"\s+", "", compile_body)
        self.assertIn(
            "past_sequence_lengths=infinicore::Tensor::empty({b},"
            "infinicore::DataType::I32",
            compact,
        )
        self.assertIn(
            "total_sequence_lengths=infinicore::Tensor::empty({b},"
            "infinicore::DataType::I32",
            compact,
        )
        self.assertIn("set_zeros(input.past_sequence_lengths.value())", compile_body)
        self.assertIn(
            "input.block_tables = infinicore::Tensor::empty({b, 1}", compile_body
        )
        self.assertIn("block_tables_vec[i] = static_cast<int32_t>(i)", compile_body)
        for dynamic_input in (
            "input_ids",
            "position_ids",
            "past_sequence_lengths",
            "total_sequence_lengths",
        ):
            self.assertIn(
                f"graph_input.{dynamic_input}.value()->copy_from(", replay_body
            )


if __name__ == "__main__":
    unittest.main()
