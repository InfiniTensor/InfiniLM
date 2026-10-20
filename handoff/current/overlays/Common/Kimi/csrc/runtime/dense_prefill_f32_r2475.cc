#include "../layers/linear/dense_prefill_f32_r2475.hpp"

// Private candidate implementation; linked once into the fresh Runtime.
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>

namespace infinilm::layers::linear {
namespace {
struct PrefillCountersR2475 {
    std::atomic<std::uint64_t> calls{0}, rows{0}, weight_bytes{0}, peak_bytes{0}, failures{0};
};
std::array<PrefillCountersR2475, 8> prefill_counters_r2475;
bool enabled_r2475() {
    const char *flag = std::getenv("INFINILM_DENSE_PREFILL_F32_R2475");
    return flag && std::strcmp(flag, "1") == 0;
}
void require_eager_r2475(infinilm::runtime::Runtime &runtime) {
    if (runtime.graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
        throw std::logic_error("r2475 prefill FP32 candidate requires eager Runtime; capture is unqualified");
    }
}
}

bool use_dense_prefill_f32_r2475(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &input,
    const infinilm::runtime::Tensor &weight,
    infinilm::runtime::DataType output_dtype) {
    using DT = infinilm::runtime::DataType;
    runtime.assertOwnerThread();
    if (!input || !weight || input->ndim() == 0 || !input->shape().back()
        || runtime.device().type() != infinilm::runtime::Device::Type::kHygon
        || input->dtype() != DT::kBFloat16 || weight->dtype() != DT::kBFloat16
        || output_dtype != DT::kBFloat16
        || input->numel() / input->shape().back() < 512 || !enabled_r2475()) {
        return false;
    }
    require_eager_r2475(runtime);
    return true;
}

infinilm::runtime::Tensor dense_prefill_f32_r2475(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &input,
    const infinilm::runtime::Tensor &weight,
    const infinilm::runtime::Tensor &output, float alpha) {
    using DT = infinilm::runtime::DataType;
    runtime.assertOwnerThread();
    if (!input || !weight || !output || input->ndim() != 2 || weight->ndim() != 2
        || output->ndim() != 2 || input->size(1) != weight->size(0)
        || output->size(0) != input->size(0) || output->size(1) != weight->size(1)
        || input->device() != runtime.device() || weight->device() != runtime.device()
        || output->device() != runtime.device()
        || !use_dense_prefill_f32_r2475(runtime, input, weight, output->dtype())) {
        throw std::invalid_argument("r2475 requires enabled eager Hygon BF16 matrix views and M>=512");
    }
    const auto device = runtime.device().index();
    if (device < 0 || device >= 8) throw std::invalid_argument("r2475 device index");
    auto &stats = prefill_counters_r2475.at(static_cast<std::size_t>(device));
    infinilm::runtime::Tensor x_source, w_source, x32, w_storage32, w32, sum32;
    try {
        x_source = input->contiguous(runtime);
        const bool transposed = weight->stride(0) == 1
            && weight->stride(1) == static_cast<std::ptrdiff_t>(weight->size(0));
        w_source = transposed ? weight->permute({1, 0})->contiguous(runtime)
                              : weight->contiguous(runtime);
        x32 = infinilm::runtime::Tensor::empty(runtime, input->shape(), DT::kFloat32);
        w_storage32 = infinilm::runtime::Tensor::empty(runtime, w_source->shape(), DT::kFloat32);
        w32 = transposed ? w_storage32->permute({1, 0}) : w_storage32;
        sum32 = infinilm::runtime::Tensor::empty(runtime, output->shape(), DT::kFloat32);
        infinilm::execution::submit<infini::ops::Cast>(runtime,
            infini::ops::Cast::DefaultConfig(runtime.device().type()), x_source, x32);
        infinilm::execution::submit<infini::ops::Cast>(runtime,
            infini::ops::Cast::DefaultConfig(runtime.device().type()), w_source, w_storage32);
        infinilm::execution::submit<infini::ops::Gemm>(runtime,
            infinilm::execution::RecordingPolicy::DeviceOnly,
            infini::ops::Gemm::DefaultConfig(runtime.device().type()), x32, w32,
            std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{alpha},
            std::optional<float>{0}, std::optional<int>{}, std::optional<int>{}, sum32);
        infinilm::execution::submit<infini::ops::Cast>(runtime,
            infini::ops::Cast::DefaultConfig(runtime.device().type()), sum32, output);
        // Enqueue counters are not GPU completion or full-model correctness proof.
        stats.calls.fetch_add(1, std::memory_order_relaxed);
        stats.rows.fetch_add(input->size(0), std::memory_order_relaxed);
        stats.weight_bytes.fetch_add(weight->numel() * 2, std::memory_order_relaxed);
        const auto bytes = std::uint64_t(x32->nbytes() + w_storage32->nbytes() + sum32->nbytes());
        auto peak = stats.peak_bytes.load(std::memory_order_relaxed);
        while (peak < bytes && !stats.peak_bytes.compare_exchange_weak(peak, bytes, std::memory_order_relaxed)) {}
        return sum32;
    } catch (...) {
        const auto failure = std::current_exception();
        stats.failures.fetch_add(1, std::memory_order_relaxed);
        // Keep every temporary alive through completion proof before unwinding.
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}
}

extern "C" int modern_dense_prefill_stats_r2475(
    int device, std::uint64_t *out, std::size_t count) {
    if (device < 0 || device >= 8 || !out || count != 5) return -1;
    auto &s = infinilm::layers::linear::prefill_counters_r2475.at(static_cast<std::size_t>(device));
    out[0] = s.calls.load(std::memory_order_relaxed);
    out[1] = s.rows.load(std::memory_order_relaxed);
    out[2] = s.weight_bytes.load(std::memory_order_relaxed);
    out[3] = s.peak_bytes.load(std::memory_order_relaxed);
    out[4] = s.failures.load(std::memory_order_relaxed);
    return 0;
}
