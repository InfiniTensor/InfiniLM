#include "infinicore/ops/moe_align.hpp"

#include "../../utils.hpp"

#ifdef ENABLE_ATEN
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <moe_align_kernel.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::moe_align_impl::aten {
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

at::Tensor ensure_i32_contig(const at::Tensor &t) {
    INFINICORE_ASSERT(t.is_contiguous());
    if (t.scalar_type() == at::kInt) {
        return t;
    }
    INFINICORE_ASSERT(t.scalar_type() == at::kLong);
    // Small [M*topk] cast — avoid ATen .to() OOR on Denglin.
    auto out = at::empty(t.sizes(), t.options().dtype(at::kInt));
    const int64_t n = t.numel();
    std::vector<int64_t> host(static_cast<size_t>(n));
    cudaMemcpy(host.data(), t.data_ptr(),
               static_cast<size_t>(n * static_cast<int64_t>(sizeof(int64_t))),
               cudaMemcpyDeviceToHost);
    std::vector<int32_t> host32(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        host32[static_cast<size_t>(i)] = static_cast<int32_t>(host[static_cast<size_t>(i)]);
    }
    cudaMemcpy(out.data_ptr(), host32.data(),
               static_cast<size_t>(n * static_cast<int64_t>(sizeof(int32_t))),
               cudaMemcpyHostToDevice);
    return out;
}

void run_moe_align_gpu(at::Tensor sorted_token_ids,
                       at::Tensor expert_ids,
                       at::Tensor num_tokens_post_padded,
                       at::Tensor topk_ids,
                       const at::Tensor *expert_map,
                       int64_t num_experts,
                       int64_t block_size,
                       bool pad_sorted_token_ids) {
    INFINICORE_ASSERT(sorted_token_ids.is_contiguous());
    INFINICORE_ASSERT(expert_ids.is_contiguous());
    INFINICORE_ASSERT(num_tokens_post_padded.is_contiguous());
    INFINICORE_ASSERT(topk_ids.is_contiguous());

    auto topk_i32 = ensure_i32_contig(topk_ids);
    at::Tensor map_i32;
    const void *map_ptr = nullptr;
    if (expert_map != nullptr) {
        INFINICORE_ASSERT(expert_map->is_contiguous());
        map_i32 = ensure_i32_contig(*expert_map);
        map_ptr = map_i32.data_ptr();
    }

    auto workspace = at::empty(
        {static_cast<int64_t>(
            ::infini::ops::qy::moe_align_workspace_bytes(
                static_cast<size_t>(num_experts)) /
            sizeof(int32_t))},
        sorted_token_ids.options().dtype(at::kInt));

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    ::infini::ops::qy::moe_align_launch(
        sorted_token_ids.data_ptr(), expert_ids.data_ptr(),
        num_tokens_post_padded.data_ptr(), topk_i32.data_ptr(), map_ptr,
        static_cast<size_t>(topk_i32.numel()),
        static_cast<size_t>(num_experts), static_cast<size_t>(block_size),
        static_cast<size_t>(sorted_token_ids.numel()),
        static_cast<size_t>(expert_ids.numel()), pad_sorted_token_ids,
        workspace.data_ptr(), reinterpret_cast<void *>(stream));
    auto err = sync_unless_capturing(stream);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("QY moe_align sync: ") +
                                 cudaGetErrorString(err));
    }
}

struct PlannedMeta {
    Tensor sorted_token_ids;
    Tensor expert_ids;
    Tensor num_tokens_post_padded;
    Tensor topk_ids;
    Tensor expert_map;
    bool has_expert_map;
    size_t num_experts;
    size_t block_size;
    bool pad_sorted_token_ids;
};

void *plan_no_map(Tensor sorted_token_ids,
                  Tensor expert_ids,
                  Tensor num_tokens_post_padded,
                  const Tensor &topk_ids,
                  const size_t num_experts,
                  const size_t block_size,
                  const bool pad_sorted_token_ids) {
    INFINICORE_ASSERT(sorted_token_ids->device().type() == Device::Type::kQy);
    return new PlannedMeta{
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        Tensor{},
        false,
        num_experts,
        block_size,
        pad_sorted_token_ids};
}

void *plan_with_map(Tensor sorted_token_ids,
                    Tensor expert_ids,
                    Tensor num_tokens_post_padded,
                    const Tensor &topk_ids,
                    const Tensor &expert_map,
                    const size_t num_experts,
                    const size_t block_size,
                    const bool pad_sorted_token_ids) {
    INFINICORE_ASSERT(sorted_token_ids->device().type() == Device::Type::kQy);
    return new PlannedMeta{
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_ids,
        expert_map,
        true,
        num_experts,
        block_size,
        pad_sorted_token_ids};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    using namespace ::infinicore::adaptor;
    set_aten_stream_to_infinicore();

    HostTimer timer("moe_align", static_cast<int64_t>(p->topk_ids->numel()));

    at::Tensor expert_map_aten;
    const at::Tensor *expert_map_ptr = nullptr;
    if (p->has_expert_map) {
        expert_map_aten = to_aten_tensor(p->expert_map);
        expert_map_ptr = &expert_map_aten;
    }

    run_moe_align_gpu(
        to_aten_tensor(p->sorted_token_ids),
        to_aten_tensor(p->expert_ids),
        to_aten_tensor(p->num_tokens_post_padded),
        to_aten_tensor(p->topk_ids),
        expert_map_ptr,
        static_cast<int64_t>(p->num_experts),
        static_cast<int64_t>(p->block_size),
        p->pad_sorted_token_ids);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

} // namespace

static bool registered = []() {
    MoeAlign::plan_dispatcher().registerDevice(Device::Type::kQy, &plan_no_map, true);
    MoeAlign::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    MoeAlign::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);

    MoeAlignWithExpertMap::plan_dispatcher().registerDevice(Device::Type::kQy, &plan_with_map, true);
    MoeAlignWithExpertMap::run_dispatcher().registerDevice(Device::Type::kQy, &run, true);
    MoeAlignWithExpertMap::cleanup_dispatcher().registerDevice(Device::Type::kQy, &cleanup, true);
    return true;
}();

} // namespace infinicore::op::moe_align_impl::aten
#endif
