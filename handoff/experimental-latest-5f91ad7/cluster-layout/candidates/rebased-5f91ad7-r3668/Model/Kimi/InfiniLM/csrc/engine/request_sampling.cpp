#include "request_sampling.hpp"

#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"

#include "../execution/submit.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinilm::engine {

// Request composition for the rank worker. RNG is supplied
// by the caller after forward; sampling remains outside graph recording.
void sample_request_token(
    infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor indices,
    infinilm::runtime::Tensor logits, float random_value,
    float top_p, int top_k, float temperature) {
    using infinilm::runtime::DataType;
    using infinilm::runtime::Device;
    using infinilm::runtime::Tensor;
    runtime.assertOwnerThread();
    INFINILM_ASSERT_TENSORS_SAME_DEVICE(indices, logits);
    if (runtime.device() != logits->device()) {
        throw std::invalid_argument("sampling runtime and tensor devices differ");
    }
    if (runtime.isGraphRecording()) {
        throw std::invalid_argument("sampling cannot be recorded before request RNG is explicit");
    }

    const auto dtype = logits->dtype();
    const auto device_type = logits->device().type();
    const bool batched = logits->ndim() == 2;
    const size_t num_rows = batched ? logits->size(0) : 1;
    const bool greedy = random_value == 0.0f || top_p == 0.0f
                     || top_k == 1 || temperature == 0.0f;
    if (greedy) {
        INFINILM_ASSERT((logits->ndim() == 1 || batched) && logits->numel() > 0);
        INFINILM_ASSERT(indices->numel() == num_rows);
        // Provider Argmax supports the common one-dimensional reduction. Keep
        // each result scalar (including B=1), matching a full-row reduction.
        auto flat_indices = indices->view({num_rows});
        for (size_t row = 0; row < num_rows; ++row) {
            auto row_logits = batched
                                ? logits->narrow({{0, row, 1}})->view({logits->size(1)})
                                : logits;
            auto row_index = flat_indices->narrow({{0, row, 1}})->view({});
            execution::submit<infini::ops::Argmax>(
                runtime, execution::RecordingPolicy::AllowHostReplay,
                infini::ops::Argmax::DefaultConfig(device_type),
                row_logits, std::optional<int64_t>{}, false, row_index);
        }
        return;
    }

    // Validate request scalars and the LM single-row/output-scalar layout.
    INFINILM_ASSERT(logits->ndim() == 1 && logits->numel() > 0 && logits->is_contiguous());
    INFINILM_ASSERT(indices->numel() == 1 && indices->is_contiguous());
    INFINILM_ASSERT(indices->dtype() == DataType::kInt32 || indices->dtype() == DataType::kInt64);
    INFINILM_ASSERT(std::isfinite(random_value));
    INFINILM_ASSERT(std::isfinite(top_p));
    INFINILM_ASSERT(top_k > 0);
    INFINILM_ASSERT(std::isfinite(temperature));
    INFINILM_ASSERT(greedy || temperature > 0.0f);

    const auto fill_config = infini::ops::Fill::DefaultConfig(device_type);
    const auto mul_config = infini::ops::Mul::DefaultConfig(device_type);
    const auto sampling_config = infini::ops::TopKTopPSamplingFromLogits::DefaultConfig(device_type);
    auto logits_2d = logits->view({1, logits->numel()});
    Tensor sampled_logits = logits_2d;
    Tensor scaled_logits;
    Tensor inverse_temperature;
    if (!greedy && temperature != 1.0f) {
        scaled_logits = Tensor::empty(runtime, logits_2d->shape(), dtype);
        inverse_temperature = Tensor::empty(runtime, {1, 1}, dtype);
        execution::submit<infini::ops::Fill>(
            runtime, execution::RecordingPolicy::AllowHostReplay, fill_config,
            inverse_temperature, static_cast<double>(1.0f / temperature), inverse_temperature);
        execution::submit<infini::ops::Mul>(
            runtime, execution::RecordingPolicy::AllowHostReplay, mul_config,
            logits_2d, inverse_temperature, scaled_logits);
        sampled_logits = scaled_logits;
    }

    // Host control views are eagerly copied into provider staging.
    // They are not Runtime-owned operands; the recording rejection above is required.
    int64_t top_k_value = greedy ? 1 : static_cast<int64_t>(top_k);
    float top_p_value = greedy ? 1.0f : top_p;
    const infini::ops::Device cpu{infini::ops::Device::Type::kCpu};
    const infini::ops::Tensor top_k_tensor(
        &top_k_value, infini::ops::Tensor::Shape{1}, infini::ops::DataType::kInt64, cpu);
    const infini::ops::Tensor top_p_tensor(
        &top_p_value, infini::ops::Tensor::Shape{1}, infini::ops::DataType::kFloat32, cpu);
    int64_t row_index = 0;
    std::optional<infini::ops::Tensor> row_indices;
    if (indices->dtype() == DataType::kInt64) {
        row_indices.emplace(
            &row_index, infini::ops::Tensor::Shape{1}, infini::ops::DataType::kInt64, cpu);
    }
    auto output = indices->as_strided({1}, {1});
    uint32_t random_bits;
    static_assert(sizeof(random_bits) == sizeof(random_value));
    std::memcpy(&random_bits, &random_value, sizeof(random_bits));
    const std::optional<int64_t> seed{
        static_cast<int64_t>(uint64_t{0x4c4d5f73616d706cULL} ^ random_bits)};
    const std::optional<int64_t> offset{0};
    execution::submit<infini::ops::TopKTopPSamplingFromLogits>(
        runtime, execution::RecordingPolicy::AllowHostReplay, sampling_config,
        sampled_logits, top_k_tensor, top_p_tensor, row_indices,
        std::string{"joint"}, true, false, seed, offset, output);
}

} // namespace infinilm::engine
