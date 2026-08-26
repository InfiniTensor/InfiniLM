#include "infinicore/ops/random_sample.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/fill.h"
#include "base/mul.h"
#include "base/top_k_top_p_sampling_from_logits.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

namespace infinicore::op::random_sample_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

constexpr uint64_t kRandomValueSeedDomain = 0x4c4d5f73616d706cULL;

int64_t seed_from_random_value(float random_value) {
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(random_value));
    std::memcpy(&bits, &random_value, sizeof(bits));
    return static_cast<int64_t>(kRandomValueSeedDomain ^ bits);
}

void calculate(
    Tensor indices, Tensor logits,
    float random_value, float top_p, int top_k, float temperature) {
    const auto dtype = logits->dtype();
    INFINICORE_ASSERT(logits->device().type() == Device::Type::kNvidia);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices, logits);
    INFINICORE_ASSERT(logits->ndim() == 1 && logits->numel() > 0 && logits->is_contiguous());
    INFINICORE_ASSERT(
        dtype == DataType::kFloat16
        || dtype == DataType::kBFloat16
        || dtype == DataType::kFloat32
        || dtype == DataType::kFloat64);
    INFINICORE_ASSERT(indices->numel() == 1 && indices->is_contiguous());
    INFINICORE_ASSERT(
        indices->dtype() == DataType::kInt32
        || indices->dtype() == DataType::kInt64);
    INFINICORE_ASSERT(std::isfinite(random_value));
    INFINICORE_ASSERT(std::isfinite(top_p));
    INFINICORE_ASSERT(top_k > 0);
    INFINICORE_ASSERT(std::isfinite(temperature));

    const bool greedy = random_value == 0.0f
                     || top_p == 0.0f
                     || top_k == 1
                     || temperature == 0.0f;
    INFINICORE_ASSERT(greedy || temperature > 0.0f);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    const TensorMeta logits_meta(logits);
    const auto device_type = logits_meta.device.type();
    auto fill_config = ::infinicore::op::infiniops::defaultConfigForDevice<infini::ops::Fill>(
        device_type);
    auto mul_config = ::infinicore::op::infiniops::defaultConfigForDevice<infini::ops::Mul>(
        device_type);
    auto sampling_config = ::infinicore::op::infiniops::defaultConfigForDevice<
        infini::ops::TopKTopPSamplingFromLogits>(device_type);

    auto logits_2d = logits->view({1, logits->numel()});
    Tensor sampled_logits = logits_2d;
    Tensor scaled_logits;
    Tensor inverse_temperature;
    if (!greedy && temperature != 1.0f) {
        scaled_logits = Tensor::empty(logits_2d->shape(), dtype, logits->device());
        inverse_temperature = Tensor::empty({1, 1}, dtype, logits->device());

        const TensorMeta inverse_temperature_meta(inverse_temperature);
        infini::ops::Fill::Call(
            handle,
            fill_config,
            inverse_temperature_meta.tensor(inverse_temperature),
            static_cast<double>(1.0f / temperature),
            inverse_temperature_meta.tensor(inverse_temperature));
        infini::ops::Mul::Call(
            handle,
            mul_config,
            TensorMeta(logits_2d).tensor(logits_2d),
            inverse_temperature_meta.tensor(inverse_temperature),
            TensorMeta(scaled_logits).tensor(scaled_logits));
        sampled_logits = scaled_logits;
    }

    int64_t top_k_value = greedy ? 1 : static_cast<int64_t>(top_k);
    float top_p_value = greedy ? 1.0f : top_p;
    const infini::ops::Device cpu{infini::ops::Device::Type::kCpu};
    const infini::ops::Tensor top_k_tensor(
        &top_k_value,
        infini::ops::Tensor::Shape{1},
        infini::ops::DataType::kInt64,
        cpu);
    const infini::ops::Tensor top_p_tensor(
        &top_p_value,
        infini::ops::Tensor::Shape{1},
        infini::ops::DataType::kFloat32,
        cpu);

    int64_t row_index = 0;
    std::optional<infini::ops::Tensor> row_indices;
    if (indices->dtype() == DataType::kInt64) {
        row_indices.emplace(
            &row_index,
            infini::ops::Tensor::Shape{1},
            infini::ops::DataType::kInt64,
            cpu);
    }

    // InfiniCore accepts a scalar result, while InfiniOps requires [batch].
    auto output = indices->as_strided({1}, {1});
    const std::optional<int64_t> seed{seed_from_random_value(random_value)};
    const std::optional<int64_t> offset{0};
    infini::ops::TopKTopPSamplingFromLogits::Call(
        handle,
        sampling_config,
        TensorMeta(sampled_logits).tensor(sampled_logits),
        top_k_tensor,
        top_p_tensor,
        row_indices,
        std::string{"joint"},
        true,
        false,
        seed,
        offset,
        TensorMeta(output).tensor(output));
}

} // namespace

static bool registered = []() {
    RandomSample::dispatcher().registerDevice(Device::Type::kNvidia, &calculate);
    return true;
}();

} // namespace infinicore::op::random_sample_impl::infiniops
#endif
