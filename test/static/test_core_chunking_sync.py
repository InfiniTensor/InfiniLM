import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def function_body(source: str, signature: str) -> str:
    signature_start = source.index(signature)
    body_start = source.index("{", signature_start)
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start : index + 1]
    raise AssertionError(f"Unterminated function body for {signature}")


class CoreChunkingSyncTest(unittest.TestCase):
    def test_paged_caching_chunks_long_prompts(self) -> None:
        source = read_source("csrc/infinicore/src/ops/paged_caching/paged_caching.cc")
        body = function_body(source, "void paged_caching_(")

        self.assertIn("MAX_TOKENS_PER_LAUNCH = 32768", body)
        self.assertIn("start += MAX_TOKENS_PER_LAUNCH", body)
        for tensor in ("k", "v", "slot_mapping"):
            with self.subTest(tensor=tensor):
                self.assertIn(f"{tensor}->narrow({{{{0, start, chunk_size}}}})", body)

    def test_swiglu_chunks_only_at_row_boundaries(self) -> None:
        source = read_source("csrc/infinicore/src/ops/swiglu/swiglu.cc")
        body = function_body(source, "void swiglu_(")

        self.assertIn("MAX_ELEMENTS_PER_LAUNCH = Size{1} << 30", body)
        self.assertIn("row_width = c->size(c->ndim() - 1)", body)
        self.assertIn("max_rows = MAX_ELEMENTS_PER_LAUNCH / row_width", body)
        for tensor in ("c_rows", "a_rows", "b_rows"):
            with self.subTest(tensor=tensor):
                self.assertIn(f"{tensor}->narrow({{{{0, start, rows}}}})", body)

    def test_static_attention_keeps_value_head_dimension(self) -> None:
        source = read_source("csrc/layers/attention/backends/static_attn.cpp")
        forward = function_body(
            source, "infinicore::Tensor StaticAttentionImpl::forward("
        )
        update = function_body(source, "StaticAttentionImpl::do_kv_cache_update(")

        self.assertIn("value_head_dim = v_reshaped->size(3)", forward)
        self.assertIn("total_seq_len, value_head_dim", forward)
        self.assertIn("num_heads_ * value_head_dim", forward)
        self.assertIn("v_cache_layer->narrow({{3, 0, value->size(3)}})", update)


if __name__ == "__main__":
    unittest.main()
