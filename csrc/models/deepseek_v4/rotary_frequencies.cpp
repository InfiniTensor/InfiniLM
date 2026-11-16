#include "rotary_frequencies.hpp"

#include "runtime/runtime.hpp"
#include "execution/submit.hpp"
#include "base/deepseek_v4_rotary_frequencies.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v4 {

runtime::Tensor make_rotary_frequencies(runtime::Runtime &runtime,
                                        const RotaryFrequencyConfig &config, bool split_cos_sin) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) {
        throw std::logic_error("DeepSeek V4 rotary tables must be prepared before graph recording");
    }
    const auto [dim, positions, theta, original_context, factor, beta_fast, beta_slow] = config;
    if (dim == 0 || dim % 2 != 0 || positions == 0
        || positions > std::numeric_limits<size_t>::max() / sizeof(float) / dim
        || !std::isfinite(theta) || theta <= 1.0 || !std::isfinite(factor) || factor < 1.0
        || !std::isfinite(beta_fast) || !std::isfinite(beta_slow)
        || beta_slow <= 0.0 || beta_fast <= beta_slow) {
        throw std::invalid_argument("Invalid DeepSeek V4 rotary frequency configuration");
    }
    if (runtime.device().type() == runtime::Device::Type::kHygon) {
        double low=0, high=0;
        if (original_context) {
            constexpr double two_pi=6.283185307179586476925286766559;
            const auto correction=[&](double rotations) {
                return double(dim)*std::log(double(original_context)/(rotations*two_pi))/(2*std::log(theta));
            };
            low=std::max(std::floor(correction(beta_fast)),0.0);
            high=std::min(std::ceil(correction(beta_slow)),double(dim-1));
        }
        auto result=runtime::Tensor::empty(runtime,{positions,dim},runtime::DataType::kFloat32);
        using Op=infini::ops::DeepseekV4RotaryFrequencies;
        execution::submit<Op>(runtime,execution::RecordingPolicy::DeviceOnly,
            Op::DefaultConfig(runtime.device().type()),result,theta,bool(original_context),low,high,factor,split_cos_sin);
        return result;
    }
    const auto half_dim = dim / 2;
    std::vector<double> frequencies(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        frequencies[i] = std::pow(theta, -static_cast<double>(2 * i) / dim);
    }
    if (original_context && factor > 1.0) {
        constexpr double two_pi = 6.283185307179586476925286766559;
        const auto correction_dim = [&](double rotations) {
            return static_cast<double>(dim) * (std::log(static_cast<double>(original_context))
                                               - std::log(rotations) - std::log(two_pi))
                 / (2.0 * std::log(theta));
        };
        const auto last = static_cast<double>(dim - 1);
        const auto low = std::clamp(std::floor(correction_dim(beta_fast)), 0.0, last);
        const auto high = std::clamp(std::ceil(correction_dim(beta_slow)), 0.0, last);
        const auto width = high == low ? 0.001 : high - low;
        for (size_t i = 0; i < half_dim; ++i) {
            const auto ramp = std::clamp((static_cast<double>(i) - low) / width, 0.0, 1.0);
            frequencies[i] *= (1.0 - ramp) + ramp / factor;
        }
    }
    std::vector<float> values(positions * dim);
    for (size_t position = 0; position < positions; ++position) {
        for (size_t i = 0; i < half_dim; ++i) {
            const auto angle = static_cast<double>(position) * frequencies[i];
            const auto row = position * dim;
            const auto cos_col = split_cos_sin ? i : 2 * i;
            const auto sin_col = split_cos_sin ? i + half_dim : 2 * i + 1;
            values[row + cos_col] = static_cast<float>(std::cos(angle));
            values[row + sin_col] = static_cast<float>(std::sin(angle));
        }
    }
    auto result = runtime::Tensor::empty(runtime, {positions, dim}, runtime::DataType::kFloat32);
    runtime.memcpyH2D(result->data(), values.data(), values.size() * sizeof(float), false);
    return result;
}

} // namespace infinilm::models::deepseek_v4
