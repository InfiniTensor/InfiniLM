import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_source(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class CoreRuntimeSyncTest(unittest.TestCase):
    def test_rope_exposes_cache_accessors(self) -> None:
        header = read_source("csrc/infinicore/include/infinicore/nn/rope.hpp")

        self.assertIn("const Tensor &sin_cache() const { return sin_cache_; }", header)
        self.assertIn("const Tensor &cos_cache() const { return cos_cache_; }", header)

    def test_parameter_accepts_degenerate_2d_transpose(self) -> None:
        source = read_source("csrc/infinicore/src/nn/parameter.cc")

        for contract in (
            "expected_shape.size() == 2",
            "actual_shape.size() == 2",
            "expected_shape[0] == actual_shape[1]",
            "expected_shape[1] == actual_shape[0]",
            "expected_shape[0] == 1 || expected_shape[1] == 1",
            "tensor->is_contiguous()",
            "tensor->contiguous()->view(expected_shape)",
            "source_tensor->narrow",
            "impl_->copy_from(source_tensor)",
        ):
            self.assertIn(contract, source)

    def test_view_error_reports_source_layout_and_target_shape(self) -> None:
        source = read_source("csrc/infinicore/src/tensor/view.cc")

        self.assertIn("old_shape=", source)
        self.assertIn("old_strides=", source)
        self.assertIn("new_shape=", source)
        self.assertEqual(source.count("incompatible_view_message("), 3)


if __name__ == "__main__":
    unittest.main()
