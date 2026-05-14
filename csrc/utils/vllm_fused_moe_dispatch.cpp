#include "vllm_fused_moe_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/tensor.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <optional>
#include <string>

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/SymInt.h>
#include <cuda_runtime.h>
#endif

namespace py = pybind11;

namespace infinilm::vllm_fused_moe_dispatch {

namespace {

// -1: unavailable (do not attempt again), 0: unknown, 1: available
std::atomic<int> g_fused_available{0};

bool env_legacy_dispatch() {
    const char *v = std::getenv("INFINILM_VLLM_FUSED_DISPATCH");
    return v != nullptr && std::string(v) == "legacy";
}

#ifdef ENABLE_ATEN
void bridge_ic_stream_to_torch_stream() {
    cudaStream_t ic_stream = cudaStream_t(infinicore::context::getStream());
    cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
    cudaEvent_t ev{};
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, ic_stream);
    cudaStreamWaitEvent(torch_stream, ev, 0);
    cudaEventDestroy(ev);
}

void bridge_torch_stream_to_ic_stream() {
    cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream().stream();
    cudaStream_t ic_stream = cudaStream_t(infinicore::context::getStream());
    cudaEvent_t ev{};
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    cudaEventRecord(ev, torch_stream);
    cudaStreamWaitEvent(ic_stream, ev, 0);
    cudaEventDestroy(ev);
}

infinicore::DataType ic_dtype_from_at(at::ScalarType st) {
    switch (st) {
    case at::kFloat:
        return infinicore::DataType::F32;
    case at::kHalf:
        return infinicore::DataType::F16;
    case at::kBFloat16:
        return infinicore::DataType::BF16;
    case at::kInt:
        return infinicore::DataType::I32;
    default:
        throw std::runtime_error("vllm_fused_moe_dispatch: unsupported ATen dtype for IC view");
    }
}

infinicore::Device ic_device_from_at(const at::Tensor &t) {
    if (t.is_cuda()) {
        return infinicore::Device(infinicore::Device::Type::NVIDIA, static_cast<size_t>(t.get_device()));
    }
    return infinicore::Device(infinicore::Device::Type::CPU, 0);
}

infinicore::Tensor ic_tensor_view_from_at(const at::Tensor &t) {
    void *ptr = t.data_ptr();
    infinicore::Shape shape(t.sizes().begin(), t.sizes().end());
    infinicore::Strides strides(t.strides().begin(), t.strides().end());
    auto dtype = ic_dtype_from_at(t.scalar_type());
    auto dev = ic_device_from_at(t);
    if (t.is_contiguous()) {
        return infinicore::Tensor::from_blob(ptr, shape, dtype, dev);
    }
    return infinicore::Tensor::strided_from_blob(ptr, shape, strides, dtype, dev);
}

// Boxed dispatcher call: schema follows vLLM 0.19 ``outplace_fused_experts`` but is
// registered under the InfiniLM fragment library (``torch.ops.infinilm.*``).
at::Tensor call_infinilm_outplace_fused_experts(
    const at::Tensor &hidden_states,
    const at::Tensor &w1,
    const at::Tensor &w2,
    const at::Tensor &topk_weights,
    const at::Tensor &topk_ids) {
    static const c10::OperatorHandle op =
        c10::Dispatcher::singleton().findSchemaOrThrow("infinilm::outplace_fused_experts", "");
    torch::jit::Stack stack;
    stack.reserve(24);
    stack.emplace_back(hidden_states);
    stack.emplace_back(w1);
    stack.emplace_back(w2);
    stack.emplace_back(topk_weights);
    stack.emplace_back(topk_ids);
    stack.emplace_back(c10::IValue("silu"));
    stack.emplace_back(false);
    stack.emplace_back(false);
    stack.emplace_back(false);
    stack.emplace_back(false);
    stack.emplace_back(false);
    stack.emplace_back(); // ocp_mx_scheme None
    stack.emplace_back(false);
    stack.emplace_back(c10::SymInt(-1));
    stack.emplace_back(); // expert_map
    stack.emplace_back(); // w1_scale
    stack.emplace_back(); // w2_scale
    stack.emplace_back(); // w1_zp
    stack.emplace_back(); // w2_zp
    stack.emplace_back(); // a1_scale
    stack.emplace_back(); // a2_scale
    stack.emplace_back(); // block_shape
    stack.emplace_back(); // w1_bias
    stack.emplace_back(); // w2_bias
    op.callBoxed(&stack);
    return stack.back().toTensor();
}

std::optional<infinicore::Tensor> try_fused_experts_dispatcher(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &w1_stacked,
    const infinicore::Tensor &w2_stacked,
    const infinicore::Tensor &topk_weights,
    const infinicore::Tensor &topk_ids) {
    try {
        if (hidden_states->device().getType() == infinicore::Device::Type::NVIDIA) {
            bridge_ic_stream_to_torch_stream();
        }

        at::Tensor h = infinicore::adaptor::to_aten_tensor(hidden_states);
        at::Tensor w1 = infinicore::adaptor::to_aten_tensor(w1_stacked);
        at::Tensor w2 = infinicore::adaptor::to_aten_tensor(w2_stacked);
        at::Tensor tw = infinicore::adaptor::to_aten_tensor(topk_weights);
        at::Tensor ids = infinicore::adaptor::to_aten_tensor(topk_ids);

        if (!h.is_contiguous()) {
            h = h.contiguous();
        }
        if (w1.stride(-1) != 1) {
            w1 = w1.contiguous();
        }
        if (w2.stride(-1) != 1) {
            w2 = w2.contiguous();
        }

        // Match vllm_fused_moe_bridge.fused_experts_ic: only the current CUDA stream,
        // not a full device sync (device sync can dominate MoE step time vs Python path).
        if (h.is_cuda()) {
            cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream());
        }

        at::Tensor out = call_infinilm_outplace_fused_experts(h, w1, w2, tw, ids);

        if (out.is_cuda()) {
            bridge_torch_stream_to_ic_stream();
        }

        return ic_tensor_view_from_at(out);
    } catch (const std::exception &e) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] dispatcher path failed: %s\n", e.what());
                std::fflush(stderr);
            }
        }
        return std::nullopt;
    } catch (...) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] dispatcher path failed: unknown\n");
                std::fflush(stderr);
            }
        }
        return std::nullopt;
    }
}
#endif

std::optional<infinicore::Tensor> try_fused_experts_python_bridge(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &w1_stacked,
    const infinicore::Tensor &w2_stacked,
    const infinicore::Tensor &topk_weights,
    const infinicore::Tensor &topk_ids) {
    try {
        py::object tensor_mod = py::module_::import("infinicore.tensor");
        py::object TensorCls = tensor_mod.attr("Tensor");
        py::object bridge = py::module_::import("infinicore.vllm_fused_moe_bridge");
        py::object fn = bridge.attr("fused_experts_ic");

        py::object h_py = TensorCls(py::cast(hidden_states));
        py::object w1_py = TensorCls(py::cast(w1_stacked));
        py::object w2_py = TensorCls(py::cast(w2_stacked));
        py::object tw_py = TensorCls(py::cast(topk_weights));
        py::object id_py = TensorCls(py::cast(topk_ids));

        py::object out_py = fn(h_py, w1_py, w2_py, tw_py, id_py);
        py::object und = out_py.attr("_underlying");
        return und.cast<infinicore::Tensor>();
    } catch (const py::error_already_set &e) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic failed: %s\n", e.what());
            }
        }
        if (std::string(e.what()).find("requires InfiniLM vendored fused MoE") != std::string::npos) {
            g_fused_available.store(-1, std::memory_order_relaxed);
        }
        PyErr_Clear();
        return std::nullopt;
    } catch (...) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic failed: unknown exception\n");
            }
        }
        return std::nullopt;
    }
}

} // namespace

bool fused_experts_ic_available() {
    int s = g_fused_available.load(std::memory_order_relaxed);
    if (s != 0) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                static bool printed = false;
                if (!printed) {
                    printed = true;
                    std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic_available cached=%d\n", s);
                    std::fflush(stderr);
                }
            }
        }
        return s > 0;
    }
    py::gil_scoped_acquire gil;
    try {
        (void)py::module_::import("infinicore.vendor.vllm_fused_moe");
        (void)py::module_::import("infinicore.vllm_fused_moe_bridge");
        g_fused_available.store(1, std::memory_order_relaxed);
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic_available probed=1\n");
                std::fflush(stderr);
            }
        }
        return true;
    } catch (const py::error_already_set &e) {
        g_fused_available.store(-1, std::memory_order_relaxed);
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic_available probed=-1: %s\n", e.what());
                std::fflush(stderr);
            }
        }
        PyErr_Clear();
        return false;
    } catch (...) {
        g_fused_available.store(-1, std::memory_order_relaxed);
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] fused_experts_ic_available probed=-1: unknown\n");
                std::fflush(stderr);
            }
        }
        return false;
    }
}

std::optional<infinicore::Tensor> try_fused_experts_ic(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &w1_stacked,
    const infinicore::Tensor &w2_stacked,
    const infinicore::Tensor &topk_weights,
    const infinicore::Tensor &topk_ids) {
    if (!fused_experts_ic_available()) {
        return std::nullopt;
    }
    py::gil_scoped_acquire gil;
#ifdef ENABLE_ATEN
    if (!env_legacy_dispatch()) {
        if (auto r = try_fused_experts_dispatcher(
                hidden_states, w1_stacked, w2_stacked, topk_weights, topk_ids)) {
            return r;
        }
        // Dispatcher failed (e.g. schema mismatch across vLLM); fall back to Python bridge.
    }
#endif
    return try_fused_experts_python_bridge(
        hidden_states, w1_stacked, w2_stacked, topk_weights, topk_ids);
}

std::optional<GroupedSigmoidTopkIcResult> try_grouped_sigmoid_topk_ic(
    const infinicore::Tensor &router_logits_f32,
    const infinicore::Tensor &e_score_correction_bias,
    size_t top_k,
    bool norm_topk_prob,
    float routed_scaling_factor,
    size_t n_group,
    size_t topk_group) {
    if (!fused_experts_ic_available()) {
        return std::nullopt;
    }
#ifdef ENABLE_ATEN
    const bool cuda_ic = router_logits_f32->device().getType() == infinicore::Device::Type::NVIDIA;
    if (cuda_ic) {
        bridge_ic_stream_to_torch_stream();
    }
#endif
    py::gil_scoped_acquire gil;
    try {
        py::object tensor_mod = py::module_::import("infinicore.tensor");
        py::object TensorCls = tensor_mod.attr("Tensor");
        py::object bridge = py::module_::import("infinicore.vllm_fused_moe_bridge");
        py::object fn = bridge.attr("grouped_sigmoid_topk_ic_cpp");
        py::object rl_py = TensorCls(py::cast(router_logits_f32));
        py::object bias_py = TensorCls(py::cast(e_score_correction_bias));
        py::object res = fn(
            rl_py,
            bias_py,
            static_cast<int>(top_k),
            norm_topk_prob,
            static_cast<double>(routed_scaling_factor),
            static_cast<int>(n_group),
            static_cast<int>(topk_group));
        py::tuple tup = res.cast<py::tuple>();
        if (tup.size() != 2) {
#ifdef ENABLE_ATEN
            if (cuda_ic) {
                bridge_torch_stream_to_ic_stream();
            }
#endif
            return std::nullopt;
        }
        infinicore::Tensor tw = tup[0].attr("_underlying").cast<infinicore::Tensor>();
        infinicore::Tensor tid = tup[1].attr("_underlying").cast<infinicore::Tensor>();
#ifdef ENABLE_ATEN
        if (cuda_ic) {
            bridge_torch_stream_to_ic_stream();
        }
#endif
        const auto &rdev = router_logits_f32->device();
        auto tw_owned = infinicore::Tensor::empty(tw->shape(), tw->dtype(), rdev);
        tw_owned->copy_from(tw);
        auto tid_owned = infinicore::Tensor::empty(tid->shape(), tid->dtype(), rdev);
        tid_owned->copy_from(tid);
        return GroupedSigmoidTopkIcResult{std::move(tw_owned), std::move(tid_owned)};
    } catch (const py::error_already_set &e) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] grouped_sigmoid_topk_ic failed: %s\n", e.what());
            }
        }
#ifdef ENABLE_ATEN
        if (cuda_ic) {
            bridge_torch_stream_to_ic_stream();
        }
#endif
        return std::nullopt;
    } catch (...) {
        if (const char *dbg = std::getenv("INFINILM_DEBUG_VLLM_FUSED_MOE")) {
            if (std::string(dbg) == "1") {
                std::fprintf(stderr, "[INFINILM_DEBUG_VLLM_FUSED_MOE] grouped_sigmoid_topk_ic failed: unknown\n");
            }
        }
#ifdef ENABLE_ATEN
        if (cuda_ic) {
            bridge_torch_stream_to_ic_stream();
        }
#endif
        return std::nullopt;
    }
}

} // namespace infinilm::vllm_fused_moe_dispatch
