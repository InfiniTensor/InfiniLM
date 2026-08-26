import json
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


class ModernInfiniCoreCompatibilityTest(unittest.TestCase):
    def test_offset_validation_uses_modern_device_and_dtype_names(self) -> None:
        source = read_source("csrc/engine/infer_engine.cpp")
        validator = function_body(source, "size_t max_length_from_offsets(")

        self.assertIn("device().type()", validator)
        self.assertIn("Device::Type::kCpu", validator)
        self.assertIn("DataType::kInt32", validator)
        for legacy in (
            "getType()",
            "Device::Type::CPU",
            "Device::cpu()",
            "DataType::I32",
        ):
            self.assertNotIn(legacy, validator)

        moe = read_source("csrc/models/qwen3_moe/qwen3_moe_experts.cpp")
        self.assertIn("infinicclSum", moe)
        self.assertNotIn("INFINICCL_SUM", moe)

    def test_paged_compile_preserves_multi_axis_position_ids(self) -> None:
        source = read_source("csrc/engine/compiler/paged_compiler.cpp")
        compile_body = function_body(source, "void PagedCompiler::compile()")

        lookup = compile_body.index('get_or<size_t>("position_id_axes", 1)')
        validation = compile_body.index("if (position_id_axes == 0)")
        allocation = compile_body.index("input.position_ids =")
        self.assertLess(lookup, validation)
        self.assertLess(validation, allocation)
        self.assertIn("position_id_axes must be positive", compile_body)
        self.assertIn("std::vector<size_t>{position_id_axes, b}", compile_body)
        self.assertIn("std::vector<size_t>{b}", compile_body)
        self.assertIn("#include <stdexcept>", source)

    def test_row_parallel_allreduce_delegates_to_quantization(self) -> None:
        row_parallel = function_body(
            read_source("csrc/layers/linear/linear.cpp"),
            "infinicore::Tensor RowParallelLinear::forward(",
        )
        self.assertIn("compute_linear_allreduce(input, communicator_)", row_parallel)
        self.assertIn("BaseLinear::forward(input)", row_parallel)

        base_linear = function_body(
            read_source("csrc/layers/linear/base_linear.cpp"),
            "infinicore::Tensor BaseLinear::compute_linear_allreduce(",
        )
        self.assertIn("quantization_->forward_allreduce(", base_linear)
        self.assertIn("params, input, has_bias_, communicator, alpha_", base_linear)

        fallback = function_body(
            read_source("csrc/layers/quantization/base_quantization.cpp"),
            "infinicore::Tensor BaseQuantization::forward_allreduce(",
        )
        forward = fallback.index("auto output = forward(params, input, false, alpha)")
        allreduce = fallback.index("distributed::allreduce_(")
        bias_guard = fallback.index("if (has_bias)")
        add_bias = fallback.index("infinicore::op::add_(")
        self.assertLess(forward, allreduce)
        self.assertLess(allreduce, bias_guard)
        self.assertLess(bias_guard, add_bias)
        self.assertIn('params.at("bias")', fallback)
        self.assertNotIn("throw", fallback)
        self.assertIn("infinicclSum", fallback)

        base_quantization = read_source(
            "csrc/layers/quantization/base_quantization.hpp"
        )
        none_quantization = read_source(
            "csrc/layers/quantization/none_quantization.hpp"
        )
        self.assertIn(
            "virtual infinicore::Tensor forward_allreduce(", base_quantization
        )
        self.assertIn(
            "class NoneQuantization : public BaseQuantization",
            none_quantization,
        )

    def test_packed_prefill_selects_last_hidden_with_supported_ops(self) -> None:
        source = read_source("csrc/layers/causal_lm_templates/text_causal_lm.hpp")
        constructor = function_body(source, "TextCausalLM(std::shared_ptr")
        forward = function_body(
            source, "Output forward(const Input &input) const override"
        )

        self.assertIn("last_token_shift_cpu", constructor)
        self.assertIn("DataType::kInt32", constructor)
        self.assertIn("last_token_shift_cpu->to(device)", constructor)
        self.assertIn("= -1;", constructor)
        self.assertIn("infinicore::Tensor last_token_shift_", source)
        self.assertNotIn("select_last_token_hidden", source)

        offsets = forward.index("input.input_offsets.value()->narrow(")
        add = forward.index("infinicore::op::add(")
        embedding = forward.index("infinicore::op::embedding(")
        lm_head = forward.index("lm_head_->forward(lm_head_input)")
        self.assertLess(offsets, add)
        self.assertLess(add, embedding)
        self.assertLess(embedding, lm_head)
        self.assertRegex(
            forward,
            r"auto end_offsets = input\.input_offsets\.value\(\)->narrow\("
            r"\s*\{\{0, 1, num_requests\}\}\s*\);",
        )
        self.assertRegex(
            forward,
            r"auto last_token_positions = infinicore::op::add\("
            r"\s*end_offsets, last_token_shift_\s*\);",
        )
        self.assertRegex(
            forward,
            r"lm_head_input = infinicore::op::embedding\("
            r"\s*last_token_positions, packed_hidden\s*\)",
        )
        self.assertIn("{hidden_states->size(1), hidden_states->size(2)}", forward)
        self.assertIn("->view({1, num_requests,", forward)

        config = json.loads(read_source("scripts/configs/infiniops_ops.json"))
        self.assertEqual(config["add"]["implementations"], [0])
        self.assertEqual(config["embedding"]["implementations"], [0])

    def test_null_compiled_graph_falls_back_to_eager_model_forward(self) -> None:
        source = read_source("csrc/engine/rank_worker.cpp")
        run = source[source.index("} else if (local_cmd == Command::RUN)") :]
        graph_lookup = run.index("compiler_->get_compiled(")
        fallback = run.index("if (!logits)")
        eager = run.index("model_->forward(model_args)")
        sampling = run.index("infinicore::op::random_sample_(")

        self.assertLess(graph_lookup, fallback)
        self.assertLess(fallback, eager)
        self.assertLess(eager, sampling)


if __name__ == "__main__":
    unittest.main()
