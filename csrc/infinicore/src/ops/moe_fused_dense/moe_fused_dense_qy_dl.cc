#if defined(ENABLE_ATEN) && defined(ENABLE_QY_API)

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/denglin_fused_moe_adaptor.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/ops/moe_fused_dense.hpp"

#include <moe_sum_kernel.h>
#include <silu_and_mul_kernel.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::moe_fused_dense_impl::qy_dl {
namespace {

bool stream_is_capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    auto st = cudaStreamIsCapturing(stream, &status);
    return st == cudaSuccess && status != cudaStreamCaptureStatusNone;
}

cudaError_t sync_unless_capturing(cudaStream_t stream) {
    if (stream_is_capturing(stream)) {
        return cudaGetLastError();
    }
    return cudaStreamSynchronize(stream);
}

cudaError_t sync_device_unless_capturing(cudaStream_t stream) {
    if (stream_is_capturing(stream)) {
        return cudaGetLastError();
    }
    return cudaDeviceSynchronize();
}

bool host_timer_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *env = std::getenv("INFINI_MOE_HOST_TIMER");
        cached = (env != nullptr && env[0] == '1' && env[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

struct HostTimer {
    const char *op;
    int64_t m;
    std::chrono::steady_clock::time_point t0;
    bool active;

    HostTimer(const char *op_name, int64_t m_val)
        : op(op_name), m(m_val), active(host_timer_enabled()) {
        if (active) {
            t0 = std::chrono::steady_clock::now();
        }
    }

    ~HostTimer() {
        if (!active) {
            return;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "host_timer op=%s ms=%.3f m=%lld\n", op, ms,
                     static_cast<long long>(m));
    }
};

} // namespace

struct PlannedMeta {
    graph::GraphTensor output;
    graph::GraphTensor hidden_states;
    graph::GraphTensor w13;
    graph::GraphTensor w2;
    graph::GraphTensor topk_weights;
    graph::GraphTensor topk_ids;
    graph::GraphTensor sorted_token_ids;
    graph::GraphTensor expert_ids;
    graph::GraphTensor num_tokens_post_padded;
};

void *plan(Tensor output,
           const Tensor &hidden_states,
           const Tensor &w13,
           const Tensor &w2,
           const Tensor &topk_weights,
           const Tensor &topk_ids,
           const Tensor &sorted_token_ids,
           const Tensor &expert_ids,
           const Tensor &num_tokens_post_padded) {
    if (!adaptor::denglin_fused_moe::available()) {
        throw std::runtime_error(
            "QY moe_fused_dense requires vllm._dl_C (set INFINICORE_DENGLIN_DL_C_SO if needed)");
    }
    return new PlannedMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(hidden_states),
        graph::GraphTensor(w13),
        graph::GraphTensor(w2),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(topk_ids),
        graph::GraphTensor(sorted_token_ids),
        graph::GraphTensor(expert_ids),
        graph::GraphTensor(num_tokens_post_padded)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    adaptor::set_aten_stream_to_infinicore();

    auto hidden = adaptor::to_aten_tensor(p->hidden_states);
    auto w13 = adaptor::to_aten_tensor(p->w13);
    auto w2 = adaptor::to_aten_tensor(p->w2);
    auto topk_weights = adaptor::to_aten_tensor(p->topk_weights);
    auto topk_ids = adaptor::to_aten_tensor(p->topk_ids);
    auto sorted_token_ids = adaptor::to_aten_tensor(p->sorted_token_ids);
    auto expert_ids = adaptor::to_aten_tensor(p->expert_ids);
    auto num_tokens_post_padded = adaptor::to_aten_tensor(p->num_tokens_post_padded);
    auto output = adaptor::to_aten_tensor(p->output);

    const int64_t m = hidden.size(0);
    const int64_t h = hidden.size(1);
    const int64_t topk = topk_ids.size(1);
    const int64_t inter2 = w13.size(1);
    const int64_t inter = inter2 / 2;

    // Workspace capacity rounding makes sorted_cap/expert_cap recovery inexact
    // (e.g. 17 instead of 16 for typical prefill). MiniCPM5 / CudaFusedMoeRunner
    // default moe_align_block_size is 16 — use that for _dl_C.
    constexpr int64_t block_size_m = 16;
    constexpr int64_t block_size_n = 64;
    constexpr int64_t block_size_k = 64;
    const std::vector<int64_t> block_shape; // empty == vLLM default

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Stage 1: hidden @ w13 -> [M, topk, 2*inter]  (vLLM fused_experts_impl)
    auto cache1 = at::empty({m, topk, inter2}, hidden.options());
    auto a1 = hidden.view({-1, h});
    auto c1 = cache1.view({-1, topk, inter2});
    {
        HostTimer timer("stage1", m);
        adaptor::denglin_fused_moe::invoke_fused_moe_opt(
            a1, w13, c1,
            std::nullopt, std::nullopt, std::nullopt,
            topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
            /*mul_routed_weight=*/false,
            topk, block_size_m, block_size_n, block_size_k,
            false, false, false, false,
            block_shape, m);
        auto err = sync_device_unless_capturing(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("QY moe_fused_dense stage1 (w13) OOR/sync: ")
                + cudaGetErrorString(err));
        }
    }

    // SiLU/GLU activation via thin QY kernel (avoids D2H/H2D and ATen OOR).
    auto flat_gate_up = cache1.reshape({m * topk, inter2}).contiguous();
    auto activated = at::empty({m * topk, inter}, hidden.options());
    {
        HostTimer timer("silu", m);
        ::infini::ops::qy::silu_and_mul_launch(
            activated.data_ptr(), flat_gate_up.data_ptr(),
            static_cast<size_t>(m * topk), static_cast<size_t>(inter),
            static_cast<int>(p->hidden_states->dtype()),
            reinterpret_cast<void *>(stream));
        auto err = sync_unless_capturing(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("QY moe_fused_dense silu sync: ") + cudaGetErrorString(err));
        }
    }

    // Stage 2: activated @ w2 -> [M, topk, H], top_k=1, mul routed weights
    auto cache3 = at::empty({m, topk, h}, hidden.options());
    auto a2 = activated.contiguous();
    auto c2 = cache3.view({-1, /*top_k=*/1, h});
    {
        HostTimer timer("stage2", m);
        adaptor::denglin_fused_moe::invoke_fused_moe_opt(
            a2, w2, c2,
            std::nullopt, std::nullopt, std::nullopt,
            topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded,
            /*mul_routed_weight=*/true,
            /*top_k=*/1, block_size_m, block_size_n, block_size_k,
            false, false, false, false,
            block_shape, m);
        auto err = sync_device_unless_capturing(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("QY moe_fused_dense stage2 (w2) OOR/sync: ")
                + cudaGetErrorString(err));
        }
    }

    // Expert combine via thin QY moe_sum (arbitrary topk).
    auto cache3_contig = cache3.is_contiguous() ? cache3 : cache3.contiguous();
    if (!output.is_contiguous()) {
        throw std::runtime_error(
            "QY moe_fused_dense combine requires contiguous output");
    }
    {
        HostTimer timer("combine", m);
        ::infini::ops::qy::moe_sum_launch(
            output.data_ptr(), cache3_contig.data_ptr(),
            static_cast<size_t>(m), static_cast<size_t>(topk),
            static_cast<size_t>(h), static_cast<int>(p->hidden_states->dtype()),
            reinterpret_cast<void *>(stream));
        auto err = sync_unless_capturing(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("QY moe_fused_dense combine sync: ") + cudaGetErrorString(err));
        }
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    // Override the ALLDEVICE infiniop registration for Denglin QY.
    MoeFusedDense::plan_dispatcher().registerDevice(Device::Type::kQy, &plan, true);
    MoeFusedDense::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    MoeFusedDense::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::moe_fused_dense_impl::qy_dl

#endif
