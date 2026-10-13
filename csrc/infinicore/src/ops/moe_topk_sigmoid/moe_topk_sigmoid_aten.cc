#include "infinicore/ops/moe_topk_sigmoid.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <moe_topk_sigmoid_kernel.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace infinicore::op::moe_topk_sigmoid_impl::aten {
namespace {

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

struct PlannedMeta {
    Tensor topk_weights;
    Tensor topk_indices;
    Tensor gating_output;
    Tensor correction_bias;
    bool renormalize;
};

void *plan(Tensor topk_weights,
           Tensor topk_indices,
           const Tensor &gating_output,
           const Tensor &correction_bias,
           const bool renormalize) {
    INFINICORE_ASSERT(topk_weights->device().type() == Device::Type::kQy);
    return new PlannedMeta{
        topk_weights,
        topk_indices,
        gating_output,
        correction_bias,
        renormalize};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    using namespace ::infinicore::adaptor;
    set_aten_stream_to_infinicore();

    auto gating_dev = to_aten_tensor(planned->gating_output);
    auto weights_dev = to_aten_tensor(planned->topk_weights);
    auto indices_dev = to_aten_tensor(planned->topk_indices);

    INFINICORE_ASSERT(gating_dev.is_contiguous());
    INFINICORE_ASSERT(weights_dev.is_contiguous());
    INFINICORE_ASSERT(indices_dev.is_contiguous());
    INFINICORE_ASSERT(gating_dev.dim() == 2);
    INFINICORE_ASSERT(weights_dev.dim() == 2);

    const auto tokens = gating_dev.size(0);
    const auto experts = gating_dev.size(1);
    const auto topk = weights_dev.size(1);
    INFINICORE_ASSERT(indices_dev.size(0) == tokens);
    INFINICORE_ASSERT(indices_dev.size(1) == topk);

    HostTimer timer("moe_topk_sigmoid", tokens);

    const void *bias_ptr = nullptr;
    int bias_dtype_tag = 0;
    if (planned->correction_bias) {
        auto bias = to_aten_tensor(planned->correction_bias);
        INFINICORE_ASSERT(bias.is_contiguous());
        INFINICORE_ASSERT(bias.numel() == experts);
        // f32/bf16/fp16 bias consumed on device (kernel converts to float).
        if (bias.scalar_type() != at::kFloat &&
            bias.scalar_type() != at::kHalf &&
            bias.scalar_type() != at::kBFloat16) {
            throw std::runtime_error(
                "QY moe_topk_sigmoid: unsupported correction_bias dtype");
        }
        bias_ptr = bias.data_ptr();
        bias_dtype_tag = static_cast<int>(planned->correction_bias->dtype());
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // Kernel writes the actual weight dtype (f16/bf16/f32) and index dtype
    // (i32/i64) — no D2H/H2D post-cast.
    ::infini::ops::qy::moe_topk_sigmoid_launch(
        weights_dev.data_ptr(), indices_dev.data_ptr(), gating_dev.data_ptr(),
        bias_ptr, static_cast<size_t>(tokens), static_cast<size_t>(experts),
        static_cast<size_t>(topk), planned->renormalize,
        static_cast<int>(planned->gating_output->dtype()),
        static_cast<int>(planned->topk_weights->dtype()),
        static_cast<int>(planned->topk_indices->dtype()), bias_dtype_tag,
        reinterpret_cast<void *>(stream));
    auto err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("QY moe_topk_sigmoid sync: ") +
                                 cudaGetErrorString(err));
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    MoeTopkSigmoid::plan_dispatcher().registerDevice(Device::Type::kQy, &plan, true);
    MoeTopkSigmoid::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    MoeTopkSigmoid::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::moe_topk_sigmoid_impl::aten
#endif
