#include "infinicore/ops/random_sample.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/fill.h"
#include "base/mul.h"
#include "base/random_sample_infinilm.h"
#include "base/top_k_top_p_sampling_from_logits.h"
#include "infinicore/context/context.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

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

float fp16_to_float(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 127 - 15 + 1;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x3ffu;
            out = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

float bf16_to_float(uint16_t h) {
    uint32_t out = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

std::vector<float> logits_to_host_f32(const Tensor &logits) {
    const auto n = logits->numel();
    const auto dtype = logits->dtype();
    std::vector<float> out(n);
    if (dtype == DataType::kFloat32) {
        context::memcpyD2H(out.data(), logits->data(), n * sizeof(float));
        return out;
    }
    if (dtype == DataType::kFloat16 || dtype == DataType::kBFloat16) {
        std::vector<uint16_t> raw(n);
        context::memcpyD2H(raw.data(), logits->data(), n * sizeof(uint16_t));
        if (dtype == DataType::kFloat16) {
            for (size_t i = 0; i < n; ++i) {
                out[i] = fp16_to_float(raw[i]);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                out[i] = bf16_to_float(raw[i]);
            }
        }
        return out;
    }
    throw std::runtime_error("random_sample host: unsupported logits dtype");
}

int32_t sample_index_host(std::vector<float> data,
                          float random_val,
                          float topp,
                          int topk,
                          float temperature) {
    const int n = static_cast<int>(data.size());
    if (n <= 0) {
        throw std::runtime_error("random_sample host: empty logits");
    }
    if (random_val == 0.0f || topp == 0.0f || topk == 1 || temperature == 0.0f) {
        return static_cast<int32_t>(
            std::distance(data.begin(), std::max_element(data.begin(), data.end())));
    }
    if (!(temperature > 0.0f)) {
        throw std::runtime_error("random_sample host: temperature must be > 0");
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return data[a] > data[b];
    });

    const int k = std::max(1, std::min(topk, n));
    const float maxv = data[order[0]];
    std::vector<float> probs(k);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp((data[order[i]] - maxv) / temperature);
        sum += probs[i];
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        return order[0];
    }
    for (int i = 0; i < k; ++i) {
        probs[i] /= sum;
    }
    float cum = 0.0f;
    float cutoff = topp;
    if (!(cutoff > 0.0f) || cutoff > 1.0f) {
        cutoff = 1.0f;
    }
    std::vector<float> cum_probs(k);
    for (int i = 0; i < k; ++i) {
        cum += probs[i];
        cum_probs[i] = cum;
    }
    const float bound = std::min(cum_probs[k - 1], cutoff) * random_val;
    int idx = 0;
    while (idx < k && cum_probs[idx] < bound) {
        ++idx;
    }
    if (idx >= k) {
        idx = k - 1;
    }
    return order[idx];
}

// MetaX without TopKTopP InfiniOps provider: D2H host sample + H2D.
void calculate_host(Tensor indices,
                    Tensor logits,
                    float random_val,
                    float topp,
                    int topk,
                    float temperature) {
    INFINICORE_ASSERT(logits->device().type() == Device::Type::kMetax);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices, logits);
    INFINICORE_ASSERT(logits->ndim() == 1 && logits->numel() > 0 && logits->is_contiguous());
    INFINICORE_ASSERT(indices->numel() == 1 && indices->is_contiguous());
    INFINICORE_ASSERT(std::isfinite(random_val));
    INFINICORE_ASSERT(std::isfinite(topp));
    INFINICORE_ASSERT(topk > 0);
    INFINICORE_ASSERT(std::isfinite(temperature));

    context::setDevice(logits->device());
    auto host_logits = logits_to_host_f32(logits);
    const int32_t idx =
        sample_index_host(std::move(host_logits), random_val, topp, topk, temperature);

    if (indices->dtype() == DataType::kInt32) {
        context::memcpyH2D(indices->data(), &idx, sizeof(int32_t), false);
        return;
    }
    if (indices->dtype() == DataType::kInt64) {
        const int64_t idx64 = static_cast<int64_t>(idx);
        context::memcpyH2D(indices->data(), &idx64, sizeof(int64_t), false);
        return;
    }
    throw std::runtime_error("random_sample host: unsupported indices dtype");
}

// Mars: on-device RandomSampleInfinilm (linked InfiniOps provider).
void calculate_mars_rsi(Tensor indices,
                        Tensor logits,
                        float random_val,
                        float topp,
                        int topk,
                        float temperature) {
    const auto dtype = logits->dtype();
    INFINICORE_ASSERT(logits->device().type() == Device::Type::kMars);
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
    INFINICORE_ASSERT(std::isfinite(random_val));
    INFINICORE_ASSERT(std::isfinite(topp));
    INFINICORE_ASSERT(topk > 0);
    INFINICORE_ASSERT(std::isfinite(temperature));

    const bool greedy = random_val == 0.0f
                     || topp == 0.0f
                     || topk == 1
                     || temperature == 0.0f;
    INFINICORE_ASSERT(greedy || temperature > 0.0f);

    context::setDevice(logits->device());
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    const TensorMeta logits_meta(logits);
    const auto device_type = logits_meta.device.type();
    auto sample_config = ::infinicore::op::infiniops::defaultConfigForDevice<
        infini::ops::RandomSampleInfinilm>(device_type);

    const float use_topp = greedy ? 0.0f : topp;
    const int64_t use_topk = greedy ? 1 : static_cast<int64_t>(topk);
    const float use_temp = greedy ? 0.0f : temperature;

    infini::ops::RandomSampleInfinilm::Call(
        handle,
        sample_config,
        logits_meta.tensor(logits),
        random_val,
        use_topp,
        use_topk,
        use_temp,
        TensorMeta(indices).tensor(indices));
}

void calculate(
    Tensor indices, Tensor logits,
    float random_value, float top_p, int top_k, float temperature) {
    const auto dtype = logits->dtype();
    INFINICORE_ASSERT(logits->device().type() == Device::Type::kNvidia || logits->device().type() == Device::Type::kMetax || logits->device().type() == Device::Type::kMars);
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
    // Nvidia: TopKTopPSamplingFromLogits. Mars: RandomSampleInfinilm (on-device).
    // MetaX without TopKTopP: host D2H fallback (not registered here).
    RandomSample::dispatcher().registerDevice(Device::Type::kNvidia, &calculate);
    RandomSample::dispatcher().registerDevice(Device::Type::kMars, &calculate_mars_rsi);
    return true;
}();

} // namespace infinicore::op::random_sample_impl::infiniops
#endif
