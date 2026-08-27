import re
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


class InfiniCoreRuntimeContractsTest(unittest.TestCase):
    def test_retained_op_headers_do_not_depend_on_legacy_public_abi(self) -> None:
        headers = (
            "bitwise_right_shift.hpp",
            "causal_conv1d.hpp",
            "chunk_gated_delta_rule.hpp",
            "gaussian_nll_loss.hpp",
            "interpolate.hpp",
            "prelu.hpp",
            "relu6.hpp",
            "recurrent_gated_delta_rule.hpp",
            "sigmoid.hpp",
        )
        for header in headers:
            with self.subTest(header=header):
                source = read_source(f"csrc/infinicore/include/infinicore/ops/{header}")
                self.assertNotIn('#include "infinicore.h"', source)
                self.assertNotRegex(source, re.compile(r"\b__export\b"))

    def test_exact_infiniops_adapter_set_is_registered(self) -> None:
        expected_registration_counts = {
            "add/add_infiniops.cc": 3,
            "add_rms_norm/add_rms_norm_infiniops.cc": 3,
            "causal_softmax/causal_softmax_infiniops.cc": 3,
            "conv2d/conv2d_infiniops.cc": 1,
            "embedding/embedding_infiniops.cc": 3,
            "gelu/gelu_infiniops.cc": 1,
            "gelutanh/gelutanh_infiniops.cc": 1,
            "gemm/gemm_infiniops.cc": 3,
            "mha_kvcache/mha_kvcache_infiniops.cc": 6,
            "mul/mul_infiniops.cc": 3,
            "multi_head_attention_varlen/mha_varlen_infiniops.cc": 6,
            "ones/ones_infiniops.cc": 3,
            "paged_caching/paged_caching_infiniops.cc": 3,
            "random_sample/random_sample_infiniops.cc": 1,
            "rearrange/rearrange_infiniops.cc": 3,
            "relu/relu_infiniops.cc": 1,
            "rms_norm/rms_norm_infiniops.cc": 3,
            "rotary_embedding/rotary_embedding_infiniops.cc": 3,
            "sigmoid/sigmoid_infiniops.cc": 3,
            "silu/silu_infiniops.cc": 1,
            "silu_and_mul/silu_and_mul_infiniops.cc": 3,
            "softmax/softmax_infiniops.cc": 1,
            "swiglu/swiglu_infiniops.cc": 3,
            "topksoftmax/topksoftmax_infiniops.cc": 3,
        }
        ops_root = ROOT / "csrc/infinicore/src/ops"
        adapters = {
            path.relative_to(ops_root).as_posix()
            for path in ops_root.glob("*/*_infiniops.cc")
        }
        self.assertEqual(adapters, set(expected_registration_counts))

        for relative_path, expected_count in expected_registration_counts.items():
            with self.subTest(adapter=relative_path):
                source = read_source(f"csrc/infinicore/src/ops/{relative_path}")
                self.assertEqual(
                    source.count("registerSupportedDevices(")
                    + source.count(".registerDevice("),
                    expected_count,
                )

        rearrange = read_source(
            "csrc/infinicore/src/ops/rearrange/rearrange_infiniops.cc"
        )
        for dispatcher in (
            "plan_dispatcher()",
            "run_dispatcher()",
            "cleanup_dispatcher()",
        ):
            self.assertIn(f"Rearrange::{dispatcher}", rearrange)
        self.assertNotIn("InfiniCore/InfiniOp implementation active", rearrange)

    def test_infiniops_adapters_do_not_use_infinilm_suffixed_apis(self) -> None:
        ops_root = ROOT / "csrc/infinicore/src/ops"
        for path in ops_root.glob("*/*_infiniops.cc"):
            with self.subTest(adapter=path.relative_to(ops_root).as_posix()):
                source = path.read_text(encoding="utf-8")
                self.assertNotRegex(
                    source, re.compile(r'#include\s+"base/[^"\n]*_infinilm\.h"')
                )
                self.assertNotRegex(
                    source, re.compile(r"infini::ops::[A-Za-z0-9_]*Infinilm")
                )

        operator_config = read_source("scripts/configs/infiniops_ops.json")
        self.assertNotIn("_infinilm", operator_config)

    def test_infiniops_adapters_select_an_active_implementation(self) -> None:
        bridge = read_source("csrc/infinicore/src/ops/infiniops_impl.hpp")
        self.assertIn("Operator::active_implementation_indices(device_type)", bridge)
        self.assertIn("implementation_indices.empty()", bridge)
        self.assertIn("implementation_indices.front()", bridge)
        self.assertIn("no active implementation for device", bridge)

        ops_root = ROOT / "csrc/infinicore/src/ops"
        adapter_paths = list(ops_root.glob("*/*_infiniops.cc"))
        adapter_paths.append(ops_root / "random_sample/random_sample.cc")
        for path in adapter_paths:
            with self.subTest(adapter=path.relative_to(ops_root).as_posix()):
                source = path.read_text(encoding="utf-8")
                if "infini::ops::" not in source or "::Call(" not in source:
                    continue
                self.assertIn("defaultConfigForDevice<", source)
                self.assertNotRegex(
                    source, re.compile(r"infini::ops::Config\s+[A-Za-z0-9_]+\s*;")
                )
                self.assertNotIn(".set_implementation_index(", source)

    def test_causal_softmax_uses_composed_infiniops_operators(self) -> None:
        source = read_source(
            "csrc/infinicore/src/ops/causal_softmax/causal_softmax_infiniops.cc"
        )

        self.assertNotIn('#include "base/causal_softmax.h"', source)
        self.assertNotIn("infini::ops::CausalSoftmax::Call", source)
        for operator in ("Fill", "Triu", "Tril", "Add", "Softmax"):
            self.assertIn(f"infini::ops::{operator}::Call", source)
        call_offsets = [
            source.index(f"infini::ops::{operator}::Call")
            for operator in ("Fill", "Triu", "Tril", "Add", "Softmax")
        ]
        self.assertEqual(call_offsets, sorted(call_offsets))
        for operator in ("Fill", "Triu", "Tril", "Add", "Softmax"):
            self.assertRegex(
                source,
                re.compile(rf"defaultConfigForDevice<\s*infini::ops::{operator}>"),
            )
        self.assertIn("total_seq_len - seq_len + 1", source)
        self.assertIn("static_cast<int64_t>(-1)", source)
        self.assertIn("output->strides() == input->strides()", source)

    def test_infiniops_adapter_temporaries_keep_owning_tensors(self) -> None:
        cases = {
            "causal_softmax/causal_softmax_infiniops.cc": (
                "std::optional<Tensor> mask_owner;",
            ),
            "paged_caching/paged_caching_infiniops.cc": (
                "Tensor scale_owner;",
                "context::memcpyH2D(scale->data(), &one, sizeof(one), false)",
            ),
            "swiglu/swiglu_infiniops.cc": ("Tensor packed_owner;",),
            "topksoftmax/topksoftmax_infiniops.cc": (
                "Tensor token_expert_indices_owner;",
            ),
        }
        for relative_path, tokens in cases.items():
            with self.subTest(adapter=relative_path):
                source = read_source(f"csrc/infinicore/src/ops/{relative_path}")
                for token in tokens:
                    self.assertIn(token, source)

    def test_gemm_call_matches_canonical_infiniops_schema(self) -> None:
        source = read_source("csrc/infinicore/src/ops/gemm/gemm_infiniops.cc")
        call = function_body(source, "void run(void *planned_meta)")

        bias = call.index("std::optional<infini::ops::Tensor>{}")
        alpha = call.index("std::optional<float>{planned->alpha}")
        beta = call.index("std::optional<float>{planned->beta}")
        self.assertLess(bias, alpha)
        self.assertLess(alpha, beta)

    def test_cat_validates_input_rank_against_reference_tensor_rank(self) -> None:
        source = read_source("csrc/infinicore/src/ops/cat/cat.cc")
        cat = function_body(source, "Tensor cat(")

        self.assertIn("int ndim = tensors[0]->ndim();", cat)
        self.assertIn(
            "tensors[i]->ndim() == ndim || tensors[i]->ndim() == 1",
            cat,
        )
        self.assertNotIn(
            "tensors[i]->ndim() == dim || tensors[i]->ndim() == 1",
            cat,
        )

    def test_qwen3_linear_bias_uses_supported_infiniops_composition(
        self,
    ) -> None:
        source = read_source("csrc/models/qwen3/qwen3_for_causal_lm.cpp")
        factory = function_body(source, "create_qwen3_model_config(")

        self.assertNotIn("linear bias is unsupported", factory)
        for field in ("attention_bias", "attention_output_bias", "mlp_bias"):
            self.assertNotIn(field, factory)

        linear = read_source("csrc/infinicore/src/ops/linear/linear.cc")
        for signature in ("void linear_(", "void linear_packed_("):
            with self.subTest(signature=signature):
                body = function_body(linear, signature)
                gemm = body.index("gemm_(")
                add = body.index("add_(out_view, out_view, bias.value())")
                self.assertIn("alpha, 0.0f", body)
                self.assertNotIn("rearrange_", body)
                self.assertLess(gemm, add)

        none_quantization = read_source(
            "csrc/layers/quantization/none_quantization.cpp"
        )
        layout = function_body(none_quantization, "NoneQuantization::get_param_layout(")
        self.assertIn("const bool tensor_parallel = split_dim >= 0", layout)
        self.assertIn("tensor_parallel ? split_dim : 0", layout)
        self.assertIn("tensor_parallel ? tp_size : 1", layout)
        self.assertIn("const bool column_parallel = split_dim == 0", layout)
        self.assertIn('{"bias", {out_features}, dtype, 0', layout)
        self.assertIn("column_parallel ? tp_size : 1", layout)

    def test_empty_operator_dispatch_throws(self) -> None:
        source = read_source(
            "csrc/infinicore/include/infinicore/ops/common/dispatcher.hpp"
        )
        lookup = function_body(source, "Fn lookup(Device::Type device_type) const")
        self.assertIn("if (fn == nullptr)", lookup)
        self.assertIn("throw std::runtime_error", lookup)
        self.assertIn("No operator implementation is registered for device", lookup)

    def test_unavailable_configurable_surfaces_are_rejected(self) -> None:
        quant_source = read_source("csrc/config/quant_config.cpp")
        quant_method = function_body(
            quant_source, "QuantConfig::get_quantization_method() const"
        )
        self.assertNotIn(
            "make_shared<infinilm::quantization::CompressedTensors>", quant_method
        )
        self.assertRegex(
            quant_method,
            re.compile(
                r'quant_method == "compressed-tensors"\)\s*\{\s*'
                r"throw std::runtime_error"
            ),
        )
        self.assertIn("end-to-end activation quantization", quant_method)
        self.assertIn("dense AWQ GEMM path", quant_method)
        self.assertIn("dense GPTQ/Marlin GEMM path", quant_method)
        self.assertIn("InfiniOps-backed MXFP4 execution path", quant_method)
        self.assertNotIn("until its kernels are available", quant_method)

        self.assertNotIn("make_shared<infinilm::quantization::MXFP4>", quant_method)
        self.assertRegex(
            quant_method,
            re.compile(
                r'quant_method == "quark"\)\s*\{\s*' r"throw std::runtime_error"
            ),
        )

        quant_header = read_source("csrc/config/quant_config.hpp")
        kv_config = function_body(
            quant_header,
            "void set_kv_quant_scheme(infinicore::DataType kv_cache_dtype)",
        )
        self.assertNotIn("KVQuantAlgo::INT8", kv_config)
        self.assertIn("throw std::runtime_error", kv_config)
        self.assertIn("KV cache INT8 quantization is unsupported", kv_config)
        self.assertIn("require FP16/BF16 KV tensors", kv_config)
        self.assertIn("no INT8 conversion path", kv_config)

        attention_config = read_source("csrc/backends/attention_backends.hpp")
        parse_backend = function_body(
            attention_config,
            "inline AttentionBackend parse_attention_backend",
        )
        self.assertRegex(
            parse_backend,
            re.compile(
                r'backend == "flash-attn"\)\s*\{\s*'
                r"return AttentionBackend::FLASH_ATTN"
            ),
        )
        self.assertNotIn("return AttentionBackend::FLASHINFER", parse_backend)
        self.assertRegex(
            parse_backend,
            re.compile(
                r'backend == "flashinfer"\)\s*\{\s*'
                r"throw std::invalid_argument"
            ),
        )
        self.assertIn(
            "selected InfiniOps provider only implements sampling", parse_backend
        )

        infer_engine = read_source("csrc/engine/infer_engine.cpp")
        flash_cache_validation = infer_engine.index(
            "attention_backend_ == backends::AttentionBackend::FLASH_ATTN"
        )
        config_creation = infer_engine.index("ConfigFactory::createConfig")
        rank_workers = infer_engine.index("RankWorker", config_creation)
        self.assertIn(
            "dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)",
            infer_engine,
        )
        self.assertIn("device_type != infinicore::Device::Type::kNvidia", infer_engine)
        self.assertIn("paged_cache_config->num_blocks() == 0", infer_engine)
        self.assertIn("paged_cache_config->block_size() == 0", infer_engine)
        self.assertIn("paged_cache_config->block_size() % 256 != 0", infer_engine)
        self.assertIn("block size divisible by 256", infer_engine)
        self.assertIn("dtype != infinicore::DataType::kFloat16", infer_engine)
        self.assertIn("dtype != infinicore::DataType::kBFloat16", infer_engine)
        self.assertIn(
            "head_dim == 0 || head_dim > 256 || head_dim % 8 != 0", infer_engine
        )
        self.assertLess(flash_cache_validation, config_creation)
        self.assertLess(flash_cache_validation, rank_workers)

    def test_modern_model_support_is_gated_before_rank_workers(self) -> None:
        source = read_source("csrc/config/config_factory.cpp")
        create = function_body(
            source,
            "std::shared_ptr<infinilm::config::ModelConfig> "
            "ConfigFactory::createConfig",
        )

        supported_model_types = {
            "baichuan",
            "chatglm",
            "fm9g",
            "fm9g7b",
            "glm4",
            "internlm3",
            "llama",
            "minicpm",
            "minicpm4",
            "qwen2",
            "qwen3",
        }
        modern_types = re.search(r"kModernModelTypes\s*\{([^}]*)\}", create)
        self.assertIsNotNone(modern_types)
        self.assertEqual(
            set(re.findall(r'"([^"]+)"', modern_types.group(1))),
            supported_model_types,
        )
        for model_type in supported_model_types:
            self.assertIn(model_type, create)

        python_engine = read_source("python/infinilm/infer_engine.py")
        self.assertIn('config_dict.get("model_type") == "minicpm"', python_engine)
        self.assertIn('config_dict["model_type"] = "minicpm4"', python_engine)
        self.assertLess(
            create.index("if (it == config_map.end())"),
            create.index("kModernModelTypes.find(model_type)"),
        )

        infer_engine = read_source("csrc/engine/infer_engine.cpp")
        config_creation = infer_engine.index("ConfigFactory::createConfig")
        rank_workers = infer_engine.index("RankWorker", config_creation)
        self.assertLess(config_creation, rank_workers)

        readme = read_source("README.md")
        self.assertIn(
            "The modern model factory enables `baichuan`, `chatglm`, `fm9g`",
            readme,
        )

        xmake = read_source("xmake.lua")
        target_start = xmake.index('target("_infinilm")')
        target_end = xmake.index("target_end()", target_start)
        model_target = xmake[target_start:target_end]
        self.assertIn(
            'add_files("csrc/**.cpp|infinicore/**.cpp|models/**.cpp")',
            model_target,
        )
        self.assertIn(
            'add_files("csrc/**.cc|infinicore/**.cc|models/**.cc")',
            model_target,
        )
        self.assertEqual(
            set(re.findall(r'"(csrc/models/[^"\n]+)"', model_target)),
            {
                "csrc/models/*.cpp",
                "csrc/models/baichuan/*.cpp",
                "csrc/models/chatglm/*.cpp",
                "csrc/models/fm9g/*.cpp",
                "csrc/models/glm4/*.cpp",
                "csrc/models/internlm3/*.cpp",
                "csrc/models/llama/*.cpp",
                "csrc/models/minicpm4/*.cpp",
                "csrc/models/qwen2/*.cpp",
                "csrc/models/qwen3/*.cpp",
            },
        )
        self.assertNotIn('add_files("csrc/**.cpp")', model_target)
        self.assertNotIn('add_files("csrc/**.cc")', model_target)
        self.assertNotIn('remove_files("csrc/infinicore/', model_target)
        self.assertIn(
            'remove_files("csrc/layers/quantization/mxfp4.cpp")', model_target
        )

        static_attention = read_source("csrc/layers/attention/backends/static_attn.cpp")
        self.assertNotIn("if (false)", static_attention)
        self.assertNotIn("op::flash_attention", static_attention)

    def test_many_variable_collectives_preflight_before_recording(self) -> None:
        cases = (
            (
                "csrc/infinicore/src/ops/distributed/allgather.cc",
                "void allgatherv_many_",
                "validate_allgatherv",
                "AllGather::execute",
            ),
            (
                "csrc/infinicore/src/ops/distributed/reduce_scatter.cc",
                "void reduce_scatterv_many_",
                "validate_reduce_scatterv",
                "ReduceScatter::execute",
            ),
        )
        loop = "for (size_t i = 0; i < inputs.size(); ++i)"
        for path, signature, validator, collective in cases:
            with self.subTest(path=path):
                source = read_source(path)
                body = function_body(source, signature)
                loop_positions = [
                    match.start() for match in re.finditer(re.escape(loop), body)
                ]
                self.assertEqual(len(loop_positions), 2)
                self.assertIn("!split_sizes.empty()", body)
                world_size = body.index("!split_sizes.empty()")
                validation = body.index(validator)
                execution = body.index(collective)
                self.assertLess(world_size, loop_positions[0])
                self.assertLess(loop_positions[0], validation)
                self.assertLess(validation, loop_positions[1])
                self.assertLess(loop_positions[1], execution)

                validator_body = function_body(source, f"void {validator}")
                for token in (
                    "dtype()",
                    "INFINICORE_ASSERT_TENSORS_SAME_DEVICE",
                    "is_contiguous()",
                    "ndim()",
                    "shape()",
                    "split_sizes",
                    "toInfinicclDataType",
                ):
                    self.assertIn(token, validator_body)

    def test_linear_packed_preserves_pretransposed_weight_layout(self) -> None:
        header = read_source("csrc/infinicore/include/infinicore/ops/linear.hpp")
        source = read_source("csrc/infinicore/src/ops/linear/linear.cc")
        packed = function_body(source, "void linear_packed_(")
        none_quantization = read_source(
            "csrc/layers/quantization/none_quantization.cpp"
        )
        split_params = function_body(
            none_quantization, "NoneQuantization::split_params("
        )

        self.assertIn("Tensor linear_packed(", header)
        self.assertIn("void linear_packed_(", header)
        self.assertIn("gemm_(", packed)
        self.assertIn("packed_weight, alpha, 0.0f", packed)
        self.assertIn("add_(out_view, out_view, bias.value())", packed)
        self.assertNotIn("packed_weight->permute", packed)
        self.assertIn("if (weight_prepacked_)", split_params)
        self.assertIn("weight_narrow_dim = 1 - narrow_dim", split_params)
        self.assertIn("static_cast<size_t>(weight_narrow_dim)", split_params)

    def test_pipeline_send_recv_is_graph_recordable_and_uses_modern_infiniccl(
        self,
    ) -> None:
        header = read_source(
            "csrc/infinicore/include/infinicore/ops/distributed/send_recv.hpp"
        )
        source = read_source("csrc/infinicore/src/ops/distributed/send_recv.cc")
        graph_header = read_source("csrc/infinicore/include/infinicore/graph/graph.hpp")
        graph_source = read_source("csrc/infinicore/src/graph/graph.cc")

        dispatch = graph_header[
            graph_header.index("#define INFINICORE_GRAPH_OP_RECORD_OR_RUN") :
        ]
        self.assertIn("context::addGraphOperator(___op)", dispatch)
        add_operator = function_body(
            graph_source, "void Graph::add_operator(std::shared_ptr<GraphOperator> op)"
        )
        self.assertIn("op_list_.push_back(op)", add_operator)
        self.assertNotIn("rejectGraphRecording", source)
        for operation in ("Send", "Recv"):
            with self.subTest(operation=operation):
                self.assertIn(
                    f"class {operation} : public graph::GraphOperator", header
                )
                class_start = header.index(f"class {operation}")
                class_end = header.index("};", class_start)
                self.assertIn(
                    "is_device_graph_capture_safe() const override { return false; }",
                    header[class_start:class_end],
                )
                constructor = function_body(source, f"{operation}::{operation}(")
                self.assertIn("validateTensor", constructor)
                self.assertIn(f"new {operation}PlannedMeta", constructor)
                self.assertNotIn(f"infiniccl{operation}(", constructor)
                execute = function_body(source, f"void {operation}::execute(")
                self.assertIn(f"INFINICORE_GRAPH_OP_RECORD_OR_RUN({operation}", execute)
                run = function_body(source, f"void {operation}::run() const")
                self.assertIn(f"run{operation}", run)
                destructor = function_body(source, f"{operation}::~{operation}()")
                self.assertIn(
                    f"delete reinterpret_cast<{operation}PlannedMeta *>", destructor
                )
                self.assertIn(f'"infiniccl{operation}"', source)
                self.assertIn(f"infiniccl{operation}(", source)
        self.assertGreaterEqual(source.count("detail::toInfinicclDataType"), 3)
        self.assertEqual(source.count("detail::checkInfiniccl("), 2)
        self.assertNotIn("infiniDtype_t", source)
        self.assertIn("void runRecv(RecvPlannedMeta &meta)", source)
        recv_run = function_body(source, "void Recv::run() const")
        self.assertIn("reinterpret_cast<RecvPlannedMeta *>(planned_meta_)", recv_run)
        self.assertNotIn("const RecvPlannedMeta", recv_run)

    def test_communication_group_initialization_is_synchronized(self) -> None:
        source = read_source("csrc/engine/distributed/communication_group.cpp")
        initializer = function_body(source, "void initializeCommunicators")
        constructor = function_body(source, "CommunicationGroup::CommunicationGroup")

        self.assertIn("initializeCommunicators(", constructor)
        self.assertIn("communicators_", constructor)
        self.assertIn("world_communicators_", constructor)
        self.assertNotIn("infinicclCommInitRank", constructor)

        for token in (
            "std::mutex start_mutex",
            "std::condition_variable start_cv",
            "bool start = false",
            "bool cancel = false",
        ):
            self.assertIn(token, initializer)

        wait = initializer.index("start_cv.wait")
        init = initializer.index("infinicclCommInitRank")
        self.assertLess(wait, init)
        self.assertIn("if (cancel)", initializer[wait:init])

        release = initializer.index("start = true", init)
        release_notify = initializer.index("start_cv.notify_all()", release)
        join = initializer.index("worker.join()", release_notify)
        self.assertLess(release, release_notify)
        self.assertLess(release_notify, join)

        failure = initializer.rindex("catch (...)")
        failure_path = initializer[failure:]
        cancel = failure_path.index("cancel = true")
        cancel_notify = failure_path.index("start_cv.notify_all()")
        cancel_join = failure_path.index("worker.join()")
        self.assertLess(cancel, cancel_notify)
        self.assertLess(cancel_notify, cancel_join)

    def test_graph_recording_is_synchronized_and_thread_owned(self) -> None:
        header = read_source("csrc/infinicore/src/graph/graph_manager.hpp")
        source = read_source("csrc/infinicore/src/graph/graph.cc")
        context_header = read_source("csrc/infinicore/src/context/context_impl.hpp")
        context_source = read_source("csrc/infinicore/src/context/context_impl.cc")

        self.assertIn("mutable std::mutex mutex_;", header)
        self.assertIn("std::thread::id capture_owner_;", header)
        for signature in (
            "bool GraphManager::is_recording() const",
            "void GraphManager::start_recording()",
            "void GraphManager::add_operator(std::shared_ptr<GraphOperator> op)",
            "std::shared_ptr<Graph> GraphManager::stop_recording()",
        ):
            body = function_body(source, signature)
            self.assertIn("std::lock_guard", body)
            self.assertIn("capture_owner_", body)

        self.assertIn(
            "static thread_local std::shared_ptr<Runtime> graph_runtime_;",
            context_header,
        )
        start = function_body(context_source, "void ContextImpl::startGraphRecording()")
        stop = function_body(
            context_source,
            "std::shared_ptr<graph::Graph> ContextImpl::stopGraphRecording()",
        )
        self.assertLess(
            start.index("current_runtime_->startGraphRecording()"),
            start.index("graph_runtime_ = current_runtime_"),
        )
        self.assertLess(
            stop.index("current_runtime_ = owner"),
            stop.index("owner->stopGraphRecording()"),
        )
        self.assertIn("current_runtime_ = previous", stop)
        self.assertIn("current_runtime_->activate()", stop)
        self.assertIn("graph_runtime_.reset()", stop)

        self.assertIn("void cancelGraphRecording() noexcept;", context_header)
        cancel = function_body(
            context_source, "void ContextImpl::cancelGraphRecording() noexcept"
        )
        self.assertIn("std::exchange(graph_runtime_, nullptr)", cancel)
        self.assertIn("owner->cancelGraphRecording()", cancel)
        self.assertIn("current_runtime_ = previous", cancel)
        self.assertNotIn("isGraphRecording()", cancel)

    def test_graph_capture_memory_is_leased_until_graph_destruction(self) -> None:
        allocator_header = read_source(
            "csrc/infinicore/src/context/allocators/pinnable_block_allocator.hpp"
        )
        allocator_source = read_source(
            "csrc/infinicore/src/context/allocators/pinnable_block_allocator.cc"
        )
        tensor_source = read_source("csrc/infinicore/src/tensor/tensor.cc")
        graph_header = read_source("csrc/infinicore/include/infinicore/graph/graph.hpp")
        runtime_source = read_source("csrc/infinicore/src/context/runtime/runtime.cc")

        self.assertIn("class PinLease", allocator_header)
        self.assertIn("size_t pin_count = 0", allocator_header)
        self.assertIn("std::shared_ptr<PinLease> commit_pin_mode()", allocator_header)
        self.assertIn("block->pin_count == 0", allocator_source)
        self.assertIn("retain_for_capture", allocator_source)
        self.assertIn("context::retainGraphMemory", tensor_source)
        self.assertIn("std::shared_ptr<void> allocation_lease_", graph_header)

        stop = function_body(
            runtime_source,
            "std::shared_ptr<graph::Graph> Runtime::stopGraphRecording()",
        )
        commit = stop.index("commit_pin_mode()")
        retain = stop.index("retain_runtime")
        finish = stop.index("finish_recording()", retain)
        self.assertLess(commit, retain)
        self.assertLess(retain, finish)

    def test_reinstantiated_blocks_have_one_free_list_entry(self) -> None:
        allocator_source = read_source(
            "csrc/infinicore/src/context/allocators/pinnable_block_allocator.cc"
        )
        mark_in_use = function_body(
            allocator_source, "size_t PinnableBlockAllocator::mark_in_use_"
        )

        self.assertIn("cls.free_blocks.erase", mark_in_use)
        self.assertIn("std::remove(", mark_in_use)
        self.assertIn("cls.free_blocks.end()", mark_in_use)

    def test_graph_destruction_synchronizes_before_releasing_leases(self) -> None:
        graph_header = read_source("csrc/infinicore/include/infinicore/graph/graph.hpp")
        graph_source = read_source("csrc/infinicore/src/graph/graph.cc")
        runtime_header = read_source("csrc/infinicore/src/context/runtime/runtime.hpp")
        runtime_source = read_source("csrc/infinicore/src/context/runtime/runtime.cc")

        self.assertIn("~Graph() noexcept", graph_header)
        destructor = function_body(graph_source, "Graph::~Graph() noexcept")
        self.assertIn("runtime_lease_->syncStreamForCleanup()", destructor)

        self.assertIn("void syncStreamForCleanup() noexcept;", runtime_header)
        self.assertIn("friend class graph::Graph;", runtime_header)
        cleanup = function_body(
            runtime_source, "void Runtime::syncStreamForCleanup() noexcept"
        )
        self.assertIn("StreamSynchronize(stream_)", cleanup)
        self.assertIn("catch (const std::exception &error)", cleanup)
        self.assertIn("catch (...)", cleanup)
        self.assertIn("restore_runtime", cleanup)

    def test_graph_compilers_cancel_capture_on_exception(self) -> None:
        context_header = read_source(
            "csrc/infinicore/include/infinicore/context/context.hpp"
        )
        runtime_header = read_source("csrc/infinicore/src/context/runtime/runtime.hpp")
        compiler_header = read_source("csrc/engine/compiler/graph_compiler.hpp")
        paged_source = read_source("csrc/engine/compiler/paged_compiler.cpp")
        static_source = read_source("csrc/engine/compiler/static_batching_compiler.cpp")

        self.assertIn("void cancelGraphRecording() noexcept;", context_header)
        self.assertIn("void cancelGraphRecording() noexcept;", runtime_header)
        guard = compiler_header[compiler_header.index("class GraphRecordingGuard") :]
        self.assertIn("context::startGraphRecording()", guard)
        self.assertIn("context::cancelGraphRecording()", guard)
        self.assertIn("context::stopGraphRecording()", guard)
        self.assertIn("~GraphRecordingGuard() noexcept", guard)
        self.assertIn("GraphRecordingGuard(GraphRecordingGuard &&) = delete", guard)
        self.assertIn(
            "GraphRecordingGuard &operator=(GraphRecordingGuard &&) = delete", guard
        )
        finish = function_body(
            compiler_header,
            "std::shared_ptr<infinicore::graph::Graph> finish()",
        )
        self.assertLess(
            finish.index("context::stopGraphRecording()"),
            finish.index("active_ = false"),
        )
        for source in (paged_source, static_source):
            self.assertIn("GraphRecordingGuard recording;", source)
            self.assertIn("auto graph = recording.finish();", source)
            self.assertNotIn("context::startGraphRecording()", source)
            self.assertNotIn("context::stopGraphRecording()", source)
        self.assertLess(
            paged_source.index("for (size_t b : decode_batch_sizes_)"),
            paged_source.index("GraphRecordingGuard recording;"),
        )

        graph_manager = function_body(
            read_source("csrc/infinicore/src/graph/graph.cc"),
            "void GraphManager::cancel_recording()",
        )
        self.assertIn("std::exchange(graph_, nullptr)", graph_manager)
        self.assertLess(
            graph_manager.index("std::lock_guard"), graph_manager.index("graph.reset()")
        )

    def test_static_graph_input_dtypes_match_scheduler_inputs(self) -> None:
        source = read_source("csrc/engine/compiler/static_batching_compiler.cpp")
        compile_body = function_body(source, "void StaticBatchingCompiler::compile()")

        expected_dtypes = {
            "input_ids": "kInt64",
            "position_ids": "kInt64",
            "past_sequence_lengths": "kInt32",
            "total_sequence_lengths": "kInt32",
            "slot_mapping": "kInt64",
        }
        for field, dtype in expected_dtypes.items():
            with self.subTest(field=field):
                self.assertRegex(
                    compile_body,
                    re.compile(
                        rf"input\.{field}\s*=\s*infinicore::Tensor::empty"
                        rf"\([^;]*DataType::{dtype},"
                    ),
                )

        self.assertIn("std::vector<int32_t> total_sequence_lengths_vec", compile_body)
        self.assertIn("b * sizeof(int32_t)", compile_body)

    def test_paged_graph_replay_updates_sequence_lengths_on_device(self) -> None:
        compiler = read_source("csrc/engine/compiler/paged_compiler.cpp")
        replay_body = function_body(
            compiler,
            "PagedCompiler::Compiled PagedCompiler::get_compiled(",
        )

        copy_expression = (
            "graph_input.total_sequence_lengths.value()->copy_from("
            "input.total_sequence_lengths.value())"
        )
        self.assertIn(copy_expression, replay_body)
        self.assertLess(
            replay_body.index(copy_expression), replay_body.index("auto graph")
        )
        self.assertNotIn("bind_host_int_array", compiler)
        self.assertNotIn("getType()", compiler)
        self.assertNotIn("Device::Type::CPU", compiler)
        self.assertNotIn("DataType::I32", compiler)

    def test_paged_decode_graph_falls_back_to_eager_under_tensor_parallelism(
        self,
    ) -> None:
        source = read_source("csrc/engine/compiler/paged_compiler.cpp")
        compile_body = function_body(source, "void PagedCompiler::compile()")
        replay_body = function_body(
            source,
            "PagedCompiler::Compiled PagedCompiler::get_compiled(",
        )

        tp_size_lookup = compile_body.index("get_tensor_model_parallel_world_size()")
        compile_guard = compile_body.index("if (tp_size > 1)")
        compile_guard_body = function_body(compile_body, "if (tp_size > 1)")
        self.assertLess(
            compile_body.index("compiled_map_decode_.clear()"), compile_guard
        )
        self.assertLess(tp_size_lookup, compile_guard)
        self.assertLess(compile_guard, compile_body.index("size_t nblocks"))
        self.assertLess(
            compile_guard, compile_body.index("GraphRecordingGuard recording")
        )
        self.assertLess(compile_guard, compile_body.index("barrier_->wait()"))
        self.assertIn("using eager decode", compile_body)
        self.assertIn("return;", compile_guard_body)

        replay_guard = replay_body.index("get_tensor_model_parallel_world_size() > 1")
        replay_guard_body = function_body(
            replay_body,
            "if (infinilm::global_state::get_tensor_model_parallel_world_size() > 1)",
        )
        self.assertLess(replay_guard, replay_body.index("size_t batch_size"))
        self.assertLess(replay_guard, replay_body.index("graph_input.input_ids"))
        self.assertIn("return {nullptr, nullptr};", replay_guard_body)

    def test_static_graph_compile_inputs_are_deterministic(self) -> None:
        source = read_source("csrc/engine/compiler/static_batching_compiler.cpp")
        compile_body = function_body(source, "void StaticBatchingCompiler::compile()")

        for field in ("input_ids", "position_ids", "past_sequence_lengths"):
            with self.subTest(field=field):
                self.assertIn(f"set_zeros(input.{field}.value())", compile_body)

        self.assertRegex(
            compile_body,
            re.compile(
                r"input\.block_tables\s*=\s*infinicore::Tensor::empty"
                r"\(\{b, 1\}, infinicore::DataType::kInt32,"
            ),
        )
        self.assertIn("block_tables_vec[i] = static_cast<int32_t>(i)", compile_body)
        self.assertIn("input.block_tables.value()->data()", compile_body)

    def test_static_graph_keeps_dynamic_cache_metadata_on_device(self) -> None:
        source = read_source("csrc/layers/attention/backends/static_attn.cpp")
        forward = function_body(
            source,
            "infinicore::Tensor StaticAttentionImpl::forward(",
        )

        self.assertIn("context::isGraphRecording()", forward)
        self.assertIn("return forward_graph_(", forward)
        self.assertIn("StaticAttentionImpl::forward_graph_", source)

        graph_forward = function_body(
            source,
            "infinicore::Tensor StaticAttentionImpl::forward_graph_(",
        )
        self.assertIn("infinicore::op::paged_caching_", graph_forward)
        self.assertIn("infinicore::op::paged_attention_", graph_forward)
        self.assertNotIn("infinicore::op::kv_caching_", graph_forward)
        self.assertNotIn("Device::Type::kCpu", graph_forward)

    def test_static_graph_uses_dynamic_canonical_cache_slots(self) -> None:
        compiler = read_source("csrc/engine/compiler/static_batching_compiler.cpp")
        compile_body = function_body(compiler, "void StaticBatchingCompiler::compile()")
        replay_body = function_body(
            compiler,
            "StaticBatchingCompiler::Compiled StaticBatchingCompiler::get_compiled(",
        )
        graph_forward = function_body(
            read_source("csrc/layers/attention/backends/static_attn.cpp"),
            "infinicore::Tensor StaticAttentionImpl::forward_graph_(",
        )

        self.assertIn("kv_cache->size(3) % 256", compiler)
        self.assertIn("kv_cache->size(1) != batch_size", compiler)
        self.assertRegex(
            compile_body,
            re.compile(
                r"input\.slot_mapping\s*=\s*infinicore::Tensor::empty"
                r"\(\{b\}, infinicore::DataType::kInt64,"
            ),
        )
        self.assertIn("i * cache_page_size", compile_body)
        self.assertIn("past_sequence_lengths[i]", replay_body)
        self.assertIn("result->second.cache_page_size", replay_body)
        self.assertIn("graph_input.slot_mapping.value()->data()", replay_body)
        self.assertIn("infinicore::op::paged_caching_", graph_forward)
        self.assertIn("attn_metadata.slot_mapping.value()", graph_forward)
        self.assertIn("permute({0, 2, 1, 3})", graph_forward)
        self.assertNotIn("infinicore::op::kv_caching_", graph_forward)

    def test_random_sampling_uses_only_canonical_infiniops(self) -> None:
        wrapper = read_source("csrc/infinicore/src/ops/random_sample/random_sample.cc")
        adapter = read_source(
            "csrc/infinicore/src/ops/random_sample/random_sample_infiniops.cc"
        )

        self.assertIn("infini::ops::Argmax::Call", wrapper)
        for sentinel in (
            "random_value != 0.0f",
            "top_p != 0.0f",
            "top_k != 1",
            "temperature != 0.0f",
        ):
            self.assertIn(sentinel, wrapper)
        self.assertIn("base/top_k_top_p_sampling_from_logits.h", adapter)
        self.assertIn("infini::ops::TopKTopPSamplingFromLogits::Call", adapter)
        for operator in ("Fill", "Mul", "TopKTopPSamplingFromLogits"):
            self.assertRegex(
                adapter,
                re.compile(rf"defaultConfigForDevice<\s*infini::ops::{operator}>"),
            )
        self.assertNotIn("set_implementation_index", adapter)
        self.assertRegex(adapter, re.compile(r"Fill::Call\(\s*handle,\s*fill_config,"))
        self.assertRegex(adapter, re.compile(r"Mul::Call\(\s*handle,\s*mul_config,"))
        self.assertRegex(
            adapter,
            re.compile(
                r"TopKTopPSamplingFromLogits::Call\(\s*handle,"
                r"\s*sampling_config,"
            ),
        )
        self.assertIn("infini::ops::Fill::Call", adapter)
        self.assertIn("infini::ops::Mul::Call", adapter)
        self.assertIn("1.0f / temperature", adapter)
        self.assertIn("const bool greedy = random_value == 0.0f", adapter)
        self.assertIn("top_k_value = greedy ? 1", adapter)
        self.assertIn("indices->dtype() == DataType::kInt64", adapter)
        self.assertIn("std::optional<infini::ops::Tensor>", adapter)
        self.assertIn('std::string{"joint"}', adapter)
        self.assertIn("indices->as_strided({1}, {1})", adapter)
        self.assertNotIn("indices->view({1})", adapter)
        self.assertNotIn("base/cast.h", adapter)
        self.assertNotIn("infini::ops::Cast", adapter)
        self.assertNotIn("RandomSampleInfinilm", adapter)
        self.assertNotIn("random_sample_infinilm.h", adapter)

    def test_canonical_rope_preserves_static_batch_positions(self) -> None:
        static_calls = (
            (
                "csrc/layers/attention/attention.cpp",
                "infinicore::Tensor Attention::forward_static_(",
            ),
            (
                "csrc/models/qwen3/qwen3_attention.cpp",
                "infinicore::Tensor Qwen3Attention::forward_static_(",
            ),
            (
                "csrc/models/qwen3_next/qwen3_next_attention.cpp",
                "infinicore::Tensor Qwen3NextAttention::forward_static_(",
            ),
        )
        for relative_path, signature in static_calls:
            with self.subTest(source=relative_path):
                body = function_body(read_source(relative_path), signature)
                self.assertIn("pos_ids_for_rope = position_ids->contiguous()", body)
                self.assertNotIn(
                    "position_ids->narrow({{0, 0, 1}})",
                    body,
                )

        deepseek = function_body(
            read_source("csrc/models/deepseek_v2/deepseek_v2_attention.cpp"),
            "infinicore::Tensor DeepseekV2Attention::forward_static_(",
        )
        self.assertIn("pos_ids = position_ids->contiguous()", deepseek)
        self.assertNotIn("position_ids_for_rope_(position_ids)", deepseek)

        rope = function_body(
            read_source("csrc/infinicore/src/nn/rope.cc"),
            "Tensor RoPE::forward(const Tensor &x,",
        )
        self.assertIn("x->ndim() == pos->ndim() + 2", rope)
        self.assertIn("x->size(x->ndim() - 1)", rope)
        self.assertIn(": head_dim_", rope)

        rope_source = read_source("csrc/infinicore/src/nn/rope.cc")
        rope_header = read_source("csrc/infinicore/include/infinicore/nn/rope.hpp")
        self.assertIn("RoPE::forward(Tensor q_out,", rope_source)
        self.assertNotIn("RoPE::forward(const Tensor &q_out,", rope_source)
        self.assertRegex(
            rope_header,
            re.compile(r"forward\(Tensor q_out,\s*Tensor k_out,"),
        )

        ernie = read_source("csrc/models/ernie4_5_vl/ernie4_5_attention.cpp")
        axis_positions = function_body(
            ernie, "infinicore::Tensor axis_positions_for_rope("
        )
        grouped = function_body(ernie, "void apply_grouped_rope_one(")
        self.assertIn(
            "position_ids->narrow({{2, axis, 1}})",
            axis_positions,
        )
        self.assertIn(
            "view({pos_shape[0], pos_shape[1]})",
            axis_positions,
        )
        self.assertIn("infinicore::op::rotary_embedding_(", grouped)
        self.assertIn("static_cast<int64_t>(group_pairs * 2)", grouped)
        self.assertIn("cos_sin_cache", grouped)
        self.assertNotIn("infinicore::op::rope_(", ernie)
        self.assertNotIn("infinicore/ops/rope.hpp", ernie)

    def test_static_graph_falls_back_for_unsupported_attention_configs(self) -> None:
        source = read_source("csrc/engine/compiler/static_batching_compiler.cpp")
        self.assertIn("bool supports_static_graph_kv_cache(", source)
        self.assertIn("bool supports_static_graph_attention(size_t batch_size)", source)
        cache_check = function_body(source, "bool supports_static_graph_kv_cache(")
        page_size = function_body(
            source, "std::optional<size_t> static_graph_cache_page_size("
        )
        capability = function_body(
            source, "bool supports_static_graph_attention(size_t batch_size)"
        )
        compile_body = function_body(source, "void StaticBatchingCompiler::compile()")

        self.assertIn("Device::Type::kNvidia", capability)
        self.assertIn(
            "static_graph_cache_page_size(batch_size).has_value()", capability
        )
        self.assertIn("get_forward_context().kv_cache_vec", page_size)
        self.assertIn("for (const auto &kv_cache : kv_cache_vec)", page_size)
        self.assertIn("kv_cache->size(1) != batch_size", page_size)
        self.assertIn("kv_cache->size(3) == 0", page_size)
        self.assertIn("kv_cache->size(3) % 256 != 0", page_size)
        self.assertIn("kv_cache->size(3) != *page_size", page_size)
        self.assertIn("kv_cache.empty()", cache_check)
        self.assertIn("kv_cache->ndim() != 5", cache_check)
        self.assertIn("kv_cache->size(0) != 2", cache_check)
        self.assertIn("kv_cache->dtype()", cache_check)
        self.assertIn("kv_cache->size(4)", cache_check)
        self.assertIn("DataType::kFloat16", cache_check)
        self.assertIn("DataType::kBFloat16", cache_check)
        self.assertIn("head_dim == 64", cache_check)
        self.assertIn("head_dim == 128", cache_check)
        self.assertIn("KVQuantAlgo::NONE", capability)
        for forbidden in ("get_dtype()", "get_kv_cache_dtype()", "get_head_dim()"):
            self.assertNotIn(forbidden, capability)
        self.assertIn("compiled_map_.clear()", compile_body)
        self.assertIn("if (!supports_static_graph_attention(b))", compile_body)
        self.assertLess(
            compile_body.index("if (!supports_static_graph_attention(b))"),
            compile_body.index("GraphRecordingGuard recording"),
        )

    def test_foreign_capture_rejects_operator_dispatch(self) -> None:
        manager_header = read_source("csrc/infinicore/src/graph/graph_manager.hpp")
        graph_header = read_source("csrc/infinicore/include/infinicore/graph/graph.hpp")
        graph_source = read_source("csrc/infinicore/src/graph/graph.cc")
        rearrange_source = read_source("csrc/infinicore/src/ops/rearrange/rearrange.cc")
        context_source = read_source("csrc/infinicore/src/context/context_impl.cc")

        for state in ("kInactive", "kActiveOwner", "kActiveNonOwner"):
            self.assertIn(state, manager_header)
        self.assertIn("CaptureState capture_state() const;", manager_header)

        is_recording = function_body(
            graph_source, "bool GraphManager::is_recording() const"
        )
        self.assertIn("kActiveNonOwner", is_recording)
        self.assertIn("throw std::runtime_error", is_recording)

        dispatch_macro = graph_header.index("INFINICORE_GRAPH_OP_RECORD_OR_RUN")
        dispatch_check = graph_header.index(
            "context::isGraphRecording()", dispatch_macro
        )
        dispatch_plan = graph_header.index("std::make_shared", dispatch_macro)
        self.assertLess(dispatch_check, dispatch_plan)

        rearrange_execute = function_body(
            rearrange_source, "void Rearrange::execute(Tensor y, const Tensor &x)"
        )
        self.assertLess(
            rearrange_execute.index("context::isGraphRecording()"),
            rearrange_execute.index("std::make_shared"),
        )

        set_device = function_body(
            context_source, "void ContextImpl::setDevice(Device device)"
        )
        self.assertNotIn("cannot switch devices during graph recording", set_device)

    def test_context_owns_one_runtime_per_thread_and_device(self) -> None:
        context_header = read_source("csrc/infinicore/src/context/context_impl.hpp")
        context_source = read_source("csrc/infinicore/src/context/context_impl.cc")
        runtime_header = read_source("csrc/infinicore/src/context/runtime/runtime.hpp")
        runtime_source = read_source("csrc/infinicore/src/context/runtime/runtime.cc")
        graph_source = read_source("csrc/infinicore/src/graph/graph.cc")

        self.assertIn("using ThreadRuntimes =", context_header)
        self.assertRegex(
            context_header,
            re.compile(
                r"std::unordered_map<std::thread::id,\s*"
                r"std::weak_ptr<Runtime>>"
            ),
        )
        self.assertIn(
            "static thread_local std::shared_ptr<Runtime> current_runtime_;",
            context_header,
        )
        create_runtime = function_body(
            context_source,
            "std::shared_ptr<Runtime> ContextImpl::getOrCreateRuntimeLocked",
        )
        self.assertIn("std::shared_ptr<Runtime>(new Runtime(device))", create_runtime)
        self.assertIn("found->second.lock()", create_runtime)
        self.assertIn("thread_id", create_runtime)
        get_current = function_body(
            context_source, "Runtime *ContextImpl::getCurrentRuntime()"
        )
        set_device = function_body(
            context_source, "void ContextImpl::setDevice(Device device)"
        )
        for body in (get_current, set_device):
            self.assertIn("std::this_thread::get_id()", body)
            self.assertIn("std::lock_guard", body)
            self.assertIn("getOrCreateRuntimeLocked", body)
        self.assertLess(
            set_device.index("current_runtime_ = std::move(runtime)"),
            set_device.rindex("current_runtime_->activate()"),
        )
        self.assertIn("kDefaultDevicePriority", get_current)

        constructor = function_body(context_source, "ContextImpl::ContextImpl()")
        for device_type in (
            "kCpu",
            "kNvidia",
            "kCambricon",
            "kAscend",
            "kMetax",
            "kMoore",
            "kIluvatar",
            "kHygon",
        ):
            self.assertIn(
                f"initializeDeviceType<Device::Type::{device_type}>()", constructor
            )

        self.assertIn("mutable std::mutex stream_mutex_;", runtime_header)
        self.assertIn(
            "mutable infini::rt::runtime::Stream stream_ = nullptr;",
            runtime_header,
        )
        self.assertNotIn("std::unordered_map<std::thread::id", runtime_header)
        self.assertIn("public std::enable_shared_from_this<Runtime>", runtime_header)

        for signature in (
            "std::shared_ptr<Memory> Runtime::allocateMemory(size_t size)",
            "std::shared_ptr<Memory> Runtime::allocatePinnedHostMemory(size_t size)",
            "std::shared_ptr<Memory> Runtime::reinstantiateBlob(std::shared_ptr<Memory> blob)",
        ):
            body = function_body(runtime_source, signature)
            self.assertIn("shared_from_this()", body)
            self.assertIn("[runtime]", body)
        allocate_memory = function_body(
            runtime_source, "std::shared_ptr<Memory> Runtime::allocateMemory"
        )
        self.assertIn("runtime->releaseDeviceMemory(p)", allocate_memory)

        release = function_body(
            runtime_source, "void Runtime::releaseDeviceMemory(std::byte *ptr) noexcept"
        )
        self.assertIn("ContextImpl::current_runtime_", release)
        sync = release.index("DeviceSynchronize")
        deallocate = release.index("device_memory_allocator_->deallocate")
        self.assertLess(sync, deallocate)

        allocator_header = read_source(
            "csrc/infinicore/src/context/allocators/pinnable_block_allocator.hpp"
        )
        allocator_source = read_source(
            "csrc/infinicore/src/context/allocators/pinnable_block_allocator.cc"
        )
        for method in (
            "void begin_pin_mode();",
            "std::shared_ptr<PinLease> commit_pin_mode();",
            "void cancel_pin_mode();",
            "void retain_for_capture(void *ptr);",
        ):
            self.assertIn(method, allocator_header)
        self.assertIn("capture_frozen_blocks_", allocator_header)
        freeze = function_body(
            allocator_source,
            "void PinnableBlockAllocator::freeze_for_capture_",
        )
        self.assertIn("capture_frozen_block_set_.insert", freeze)
        self.assertIn("++block->pin_count", freeze)
        self.assertIn("capture_frozen_blocks_.push_back(block)", freeze)
        commit_pin = function_body(
            allocator_source, "PinnableBlockAllocator::commit_pin_mode()"
        )
        self.assertIn("blocks = std::move(capture_frozen_blocks_)", commit_pin)
        self.assertIn("pinned_mode_ = false", commit_pin)
        cancel_pin = function_body(
            allocator_source, "void PinnableBlockAllocator::cancel_pin_mode()"
        )
        self.assertIn("--block->pin_count", cancel_pin)
        self.assertIn("capture_frozen_blocks_.clear()", cancel_pin)

        cancel_graph = function_body(
            runtime_source, "void Runtime::cancelGraphRecording() noexcept"
        )
        self.assertIn("graph_manager_->cancel_recording()", cancel_graph)
        self.assertIn("device_memory_allocator_->cancel_pin_mode()", cancel_graph)
        stop_graph = function_body(
            runtime_source,
            "std::shared_ptr<graph::Graph> Runtime::stopGraphRecording()",
        )
        failure_path = stop_graph[stop_graph.index("catch (...)") :]
        self.assertIn("device_memory_allocator_->cancel_pin_mode()", failure_path)
        self.assertIn("device_memory_allocator_->commit_pin_mode()", stop_graph)

        runtime_constructor = function_body(
            runtime_source, "Runtime::Runtime(Device device)"
        )
        self.assertNotIn("StreamCreate", runtime_constructor)
        stream = function_body(
            runtime_source, "infini::rt::runtime::Stream Runtime::stream() const"
        )
        for token in ("stream_mutex_", "StreamCreate", "stream_"):
            self.assertIn(token, stream)
        self.assertIn("if (stream_ == nullptr)", stream)
        self.assertIn("return stream_", stream)

        destructor = function_body(runtime_source, "Runtime::~Runtime() noexcept")
        self.assertIn("stream_mutex_", destructor)
        self.assertIn("stream_", destructor)
        self.assertIn("StreamDestroy", destructor)

        for signature in (
            "void Runtime::syncStream()",
            "void Runtime::memcpyH2D(void *dst, const void *src, size_t size, bool async)",
            "void Runtime::memcpyD2H(void *dst, const void *src, size_t size)",
            "void Runtime::memcpyD2D(void *dst, const void *src, size_t size, bool async)",
            "void Runtime::setDeviceMemory(void *ptr, int value, size_t count)",
            "void Runtime::setDeviceMemoryAsync(void *ptr, int value, size_t count, infini::rt::runtime::Stream stream)",
            "void Runtime::recordEvent(infini::rt::runtime::Event event, infini::rt::runtime::Stream stream)",
            "void Runtime::streamWaitEvent(infini::rt::runtime::Stream stream, infini::rt::runtime::Event event)",
        ):
            self.assertIn("stream()", function_body(runtime_source, signature))

        for signature in (
            "void Runtime::memcpyH2D(void *dst, const void *src, size_t size, bool async)",
            "void Runtime::memcpyD2H(void *dst, const void *src, size_t size)",
            "void Runtime::memcpyD2D(void *dst, const void *src, size_t size, bool async)",
        ):
            body = function_body(runtime_source, signature)
            self.assertIn("MemcpyAsync", body)
            self.assertIn("StreamSynchronize(current_stream)", body)
        set_memory = function_body(
            runtime_source,
            "void Runtime::setDeviceMemory(void *ptr, int value, size_t count)",
        )
        self.assertIn("MemsetAsync", set_memory)
        self.assertIn("StreamSynchronize(current_stream)", set_memory)

        instantiate = function_body(graph_source, "void Graph::instantiate()")
        self.assertIn("device_graph_->stream = context::getStream()", instantiate)

    def test_pinned_host_memory_remains_cpu_addressed(self) -> None:
        source = read_source("csrc/infinicore/src/context/runtime/runtime.cc")
        allocate_pinned = function_body(
            source,
            "std::shared_ptr<Memory> Runtime::allocatePinnedHostMemory(size_t size)",
        )

        self.assertIn("Device{Device::Type::kCpu}", allocate_pinned)
        self.assertNotIn("data_ptr, size, device_", allocate_pinned)

    def test_cross_device_copy_stages_through_host_memory(self) -> None:
        source = read_source("csrc/infinicore/src/tensor/copy.cc")
        copy_from = function_body(source, "void TensorImpl::copy_from(Tensor src)")

        self.assertIn("auto host_staging = Tensor::empty(", copy_from)
        self.assertIn("Device{Device::Type::kCpu}", copy_from)
        self.assertIn("host_staging->copy_from(src)", copy_from)
        self.assertIn("this->copy_from(host_staging)", copy_from)

    def test_cpu_copy_does_not_require_an_accelerator_operator(self) -> None:
        source = read_source("csrc/infinicore/src/tensor/copy.cc")
        copy_from = function_body(source, "void TensorImpl::copy_from(Tensor src)")

        self.assertIn("void copyCpuStrided(", source)
        self.assertIn("Cannot copy from tensor with different dtype", copy_from)
        cpu_copy = copy_from.index("if (this->device().type() == Device::Type::kCpu)")
        accelerator_copy = copy_from.index("op::rearrange_", cpu_copy)
        self.assertLess(copy_from.index("copyCpuStrided(", cpu_copy), accelerator_copy)

    def test_native_infini_rt_graph_runtime_is_enabled(self) -> None:
        xmake = read_source("xmake.lua")
        target_start = xmake.index('target("infinicore_runtime")')
        target_end = xmake.index("target_end()", target_start)
        runtime_target = xmake[target_start:target_end]

        self.assertIn("USE_INFINIRT_GRAPH", runtime_target)

        graph_header = read_source("csrc/infinicore/include/infinicore/graph/graph.hpp")
        runtime_lease = graph_header.index("runtime_lease_")
        allocation_lease = graph_header.index("allocation_lease_")
        operators = graph_header.index("op_list_")
        segments = graph_header.index("segments_")
        self.assertLess(runtime_lease, operators)
        self.assertLess(allocation_lease, operators)
        self.assertLess(runtime_lease, segments)
        self.assertLess(allocation_lease, segments)

        segment_start = graph_header.index("struct Segment {")
        segment_end = graph_header.index("segments_", segment_start)
        segment = graph_header[segment_start:segment_end]
        self.assertLess(segment.index("ops;"), segment.index("device_graph_;"))
        self.assertIn("friend class ::infinicore::Runtime", graph_header)

        runtime_source = read_source("csrc/infinicore/src/context/runtime/runtime.cc")
        stop = function_body(
            runtime_source,
            "std::shared_ptr<graph::Graph> Runtime::stopGraphRecording()",
        )
        self.assertLess(
            stop.index("device_memory_allocator_->commit_pin_mode()"),
            stop.index("graph->retain_runtime(shared_from_this(),"),
        )
        self.assertLess(
            stop.index("graph->retain_runtime(shared_from_this(),"),
            stop.rindex("graph_manager_->finish_recording()"),
        )

        graph_source = read_source("csrc/infinicore/src/graph/graph.cc")
        manager_stop = function_body(
            graph_source,
            "std::shared_ptr<Graph> GraphManager::stop_recording()",
        )
        self.assertLess(
            manager_stop.index("std::exchange(graph_, nullptr)"),
            manager_stop.index("graph->instantiate()"),
        )

    def test_h2d_copy_does_not_outlive_its_host_source(self) -> None:
        source = read_source("csrc/infinicore/src/tensor/copy.cc")
        copy_from = function_body(source, "void TensorImpl::copy_from(Tensor src)")

        self.assertNotIn("const bool async", copy_from)
        self.assertIn(
            "context::memcpyH2D(this->data(), src->data(), copy_size, false)",
            copy_from,
        )
        self.assertIn(
            "context::memcpyH2D(local_src->data(), src->data(), copy_size, false)",
            copy_from,
        )

    def test_runtime_table_lazy_initialization_is_synchronized(self) -> None:
        header = read_source("csrc/infinicore/src/context/context_impl.hpp")
        source = read_source("csrc/infinicore/src/context/context_impl.cc")

        self.assertIn("mutable std::mutex runtime_table_mutex_;", header)
        for signature in (
            "Runtime *ContextImpl::getCurrentRuntime()",
            "void ContextImpl::setDevice(Device device)",
            "size_t ContextImpl::getDeviceCount(Device::Type type)",
        ):
            self.assertIn("std::lock_guard", function_body(source, signature))

    def test_allocator_pin_transaction_is_synchronized(self) -> None:
        header = read_source(
            "csrc/infinicore/src/context/allocators/pinnable_block_allocator.hpp"
        )
        source = read_source(
            "csrc/infinicore/src/context/allocators/pinnable_block_allocator.cc"
        )

        self.assertIn("std::thread::id pin_owner_;", header)
        for signature in (
            "void PinnableBlockAllocator::begin_pin_mode()",
            "PinnableBlockAllocator::commit_pin_mode()",
            "void PinnableBlockAllocator::cancel_pin_mode()",
        ):
            transaction = function_body(source, signature)
            self.assertIn("std::lock_guard", transaction)
            self.assertIn("std::this_thread::get_id()", transaction)
        allocate = function_body(
            source, "std::byte *PinnableBlockAllocator::allocate(size_t size)"
        )
        self.assertIn("freeze_for_capture_(block)", allocate)

    def test_pinned_host_allocator_queue_is_synchronized(self) -> None:
        header = read_source(
            "csrc/infinicore/src/context/allocators/device_pinned_allocator.hpp"
        )
        source = read_source(
            "csrc/infinicore/src/context/allocators/device_pinned_allocator.cc"
        )

        self.assertIn("std::mutex gc_mutex_;", header)
        deallocate = function_body(
            source, "void DevicePinnedHostAllocator::deallocate(std::byte *ptr)"
        )
        gc = function_body(source, "void DevicePinnedHostAllocator::gc()")
        self.assertIn("std::lock_guard", deallocate)
        self.assertIn("std::lock_guard", gc)

    def test_runtime_destructor_is_nonthrowing_and_dependency_ordered(self) -> None:
        header = read_source("csrc/infinicore/src/context/runtime/runtime.hpp")
        source = read_source("csrc/infinicore/src/context/runtime/runtime.cc")

        self.assertIn("~Runtime() noexcept;", header)
        destructor = function_body(source, "Runtime::~Runtime() noexcept")
        self.assertIn("warn_runtime_cleanup_failure", destructor)
        self.assertIn("infini::rt::runtime::SetDevice", destructor)
        self.assertIn("infini::rt::runtime::StreamSynchronize", destructor)
        self.assertIn("infini::rt::runtime::StreamDestroy", destructor)
        ordered_cleanup = (
            "infini::rt::runtime::StreamSynchronize",
            "graph_manager_.reset()",
            "pinned_host_memory_allocator_.reset()",
            "device_memory_allocator_.reset()",
            "infini::rt::runtime::StreamDestroy",
        )
        positions = [destructor.index(token) for token in ordered_cleanup]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("restore_runtime", destructor)

    def test_tensor_view_requires_mutable_tensor(self) -> None:
        header = read_source("csrc/infinicore/include/infinicore/tensor.hpp")
        source = read_source("csrc/infinicore/src/tensor/tensor.cc")

        self.assertIn("infini::rt::TensorView view();", header)
        self.assertNotIn("infini::rt::TensorView view() const;", header)
        self.assertIn("infini::rt::TensorView TensorImpl::view()", source)
        self.assertNotIn("infini::rt::TensorView TensorImpl::view() const", source)


if __name__ == "__main__":
    unittest.main()
