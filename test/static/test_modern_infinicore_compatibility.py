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

    def test_current_infiniops_backend_bridges_are_registered(self) -> None:
        bridge = read_source("csrc/infinicore/src/ops/infiniops_impl.hpp")
        for device in ("kCambricon", "kAscend"):
            with self.subTest(device=device):
                self.assertIn(f"case Device::Type::{device}:", bridge)
                self.assertIn(
                    f"dispatcher.registerDevice(Device::Type::{device}, function)",
                    bridge,
                )

        rope = read_source("csrc/infinicore/src/nn/rope.cc")
        rotary = read_source(
            "csrc/infinicore/src/ops/rotary_embedding/rotary_embedding_infiniops.cc"
        )
        for device in (
            "kNvidia",
            "kMetax",
            "kIluvatar",
            "kMoore",
            "kCambricon",
            "kAscend",
        ):
            with self.subTest(device=device):
                self.assertIn(f"device_.type() == Device::Type::{device}", rope)
        self.assertIn("infiniops::isSupportedDevice(device_type)", rotary)
        self.assertEqual(rotary.count("registerSupportedDevices("), 3)

        sampling = read_source("csrc/infinicore/src/ops/random_sample/random_sample.cc")
        self.assertIn("defaultConfigForDevice<infini::ops::Argmax>", sampling)
        self.assertNotIn("set_implementation_index", sampling)
        for device in ("kMetax", "kIluvatar", "kMoore", "kCambricon", "kAscend"):
            self.assertIn(f"device_type != Device::Type::{device}", sampling)

        for relative_path in (
            "csrc/infinicore/src/ops/mha_kvcache/mha_kvcache_infiniops.cc",
            "csrc/infinicore/src/ops/multi_head_attention_varlen/"
            "mha_varlen_infiniops.cc",
        ):
            attention = read_source(relative_path)
            for device in ("kMetax", "kMoore"):
                with self.subTest(adapter=relative_path, device=device):
                    self.assertIn(f"device_type != Device::Type::{device}", attention)
                    self.assertEqual(
                        attention.count(f"Device::Type::{device}, &"),
                        3,
                    )
            self.assertIn("configForImplementation<", attention)
            self.assertIn(
                "device_type == infini::ops::Device::Type::kMoore ? 8 : 16",
                attention,
            )
            self.assertIn("device_type == Device::Type::kMoore", attention)
            self.assertIn("!= 64", attention)
            self.assertIn("!= 128", attention)
            self.assertIn("alibi_slopes.value()->ndim() != 1", attention)

        varlen_attention = read_source(
            "csrc/infinicore/src/ops/multi_head_attention_varlen/"
            "mha_varlen_infiniops.cc"
        )
        self.assertIn(
            "&& (!paged || alibi_slopes.value()->ndim() != 1)",
            varlen_attention,
        )
        self.assertIn("static_cast<std::size_t>(max_seqlen_k)", varlen_attention)
        self.assertIn("> block_table.value()->size(1) * k->size(1)", varlen_attention)

        decode_attention = read_source(
            "csrc/infinicore/src/ops/mha_kvcache/mha_kvcache_infiniops.cc"
        )
        self.assertIn("k_cache->size(0) == 0", decode_attention)

        self.assertIn("configForImplementation", bridge)
        self.assertIn("implementation_indices.end()", bridge)
        self.assertIn("is not active for device", bridge)

        engine = read_source("csrc/engine/infer_engine.cpp")
        self.assertIn(
            "flash-attn is only available on NVIDIA, MetaX, and Moore devices",
            engine,
        )

    def test_dead_moore_flash_attention_bridges_are_removed(self) -> None:
        for relative_path in (
            "csrc/infinicore/src/ops/mha_kvcache/mha_kvcache_flashattn_moore.cc",
            "csrc/infinicore/src/ops/multi_head_attention_varlen/"
            "mha_varlen_flashattn_moore.cc",
        ):
            with self.subTest(relative_path=relative_path):
                self.assertFalse((ROOT / relative_path).exists())

    def test_infini_devices_keep_their_modern_platform_names(self) -> None:
        for relative_path in (
            "python/infinilm/base_config.py",
            "test/bench/backends/infinilm.py",
        ):
            source = read_source(relative_path)
            for device in ("metax", "iluvatar", "hygon"):
                with self.subTest(relative_path=relative_path, device=device):
                    self.assertIn(f'"{device}": "{device}"', source)
                    self.assertNotIn(f'"{device}": "cuda"', source)

    def test_vendor_sdk_headers_are_available_to_infinicore_build(self) -> None:
        xmake = read_source("xmake.lua")
        for environment, default_root in (
            ("NEUWARE_HOME", "/usr/local/neuware"),
            ("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest"),
        ):
            with self.subTest(environment=environment):
                self.assertIn(f'os.getenv("{environment}")', xmake)
                self.assertIn(default_root, xmake)

        self.assertIn('add_includedirs(NEUWARE_ROOT .. "/include")', xmake)
        self.assertIn('add_linkdirs(NEUWARE_ROOT .. "/lib64")', xmake)
        self.assertIn('add_includedirs(ASCEND_ROOT .. "/include")', xmake)
        self.assertIn('add_linkdirs(ASCEND_ROOT .. "/lib64")', xmake)

    def test_infiniops_adapters_follow_the_installed_ops_closure(self) -> None:
        xmake = read_source("xmake.lua")
        self.assertIn(
            'INFINI_ROOT .. "/include/infini/operator_call_instantiations.h"',
            xmake,
        )
        self.assertIn('os.files("csrc/infinicore/src/ops/*/*_infiniops.cc")', xmake)
        self.assertIn('source:gmatch(\'#include%s+"base/([^"]+)%.h"\')', xmake)
        self.assertIn('source:gmatch("infini::ops::([%w_]+)::Call")', xmake)
        self.assertIn('"Operator<::infini::ops::" .. operator_type .. ">"', xmake)
        self.assertNotIn('"Operator<" .. operator_type .. ">"', xmake)
        self.assertIn(
            "local operator_calls = io.readfile(OPERATOR_CALLS_HEADER)", xmake
        )
        self.assertIn("local source = io.readfile(sourcefile)", xmake)
        self.assertNotIn('import("core.base.io")', xmake)
        self.assertIn('target:remove("files", sourcefile)', xmake)


if __name__ == "__main__":
    unittest.main()
