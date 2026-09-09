#include "infinicore/nn/rope_scaling_configs.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinicore::nn {

namespace {
// Define a portable PI constant to avoid relying on the non-standard M_PI macro
// which is missing on MSVC (Windows) by default.
constexpr float kPi = 3.14159265358979323846f;
} // anonymous namespace

// LongRopeScalingConfig Implementation
LongRopeScalingConfig::LongRopeScalingConfig(
    std::vector<float> short_factor,
    std::vector<float> long_factor,
    size_t original_max_position_embeddings,
    float factor)
    : short_factor_(std::move(short_factor)),
      long_factor_(std::move(long_factor)),
      original_max_position_embeddings_(original_max_position_embeddings),
      factor_(factor == 1.0f ? 1.0f : std::sqrt(1 + std::log(factor) / std::log(original_max_position_embeddings))) {}

float LongRopeScalingConfig::get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
    float _ext = (pos < original_max_position_embeddings_) ? short_factor_[dim_idx] : long_factor_[dim_idx];
    // The base inv_freq is multiplied by this scale.
    // Original: inv_freq = 1.0f / (_ext * pow(theta, 2j/head_dim))
    // New: inv_freq = base_inv_freq * (1.0f / _ext)
    return 1.0f / _ext;
}

float LongRopeScalingConfig::get_magnitude_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
    return factor_;
}

// Llama3RopeScalingConfig Implementation
Llama3RopeScalingConfig::Llama3RopeScalingConfig(
    float factor,
    float low_freq_factor,
    float high_freq_factor,
    size_t original_max_position_embeddings)
    : factor_(factor),
      low_freq_factor_(low_freq_factor),
      high_freq_factor_(high_freq_factor),
      original_max_position_embeddings_(original_max_position_embeddings) {}

float Llama3RopeScalingConfig::get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
    // Calculate the wavelength corresponding to the current inverse frequency
    float wavelen = 2.0f * static_cast<float>(kPi) / base_inv_freq;

    // Compute the wavelength thresholds that separate high, mid, and low frequencies
    float low_freq_wavelen = static_cast<float>(original_max_position_embeddings_) / low_freq_factor_;
    float high_freq_wavelen = static_cast<float>(original_max_position_embeddings_) / high_freq_factor_;

    float scale = 1.0f;

    if (wavelen < low_freq_wavelen) {
        // High-frequency band: short wavelengths retain the original scale
        scale = 1.0f;
    } else if (wavelen > high_freq_wavelen) {
        // Low-frequency band: long wavelengths are directly scaled by the factor
        scale = factor_;
    } else {
        // Mid-frequency band: apply smooth linear interpolation between 1.0 and factor_
        float smooth = (static_cast<float>(original_max_position_embeddings_) / wavelen - low_freq_factor_) / (high_freq_factor_ - low_freq_factor_);
        scale = 1.0f - smooth + smooth * factor_;
    }

    // The framework applies the scale multiplicatively (inv_freq = base_inv_freq * return_value).
    // Since the Llama3 logic divides the frequency (inv_freq = base_inv_freq / scale),
    // we return the inverse of the computed scale.
    return 1.0f / scale;
}

namespace {

float yarn_find_correction_dim(
    int num_rotations,
    size_t rotary_dim,
    float base,
    size_t original_max_position_embeddings) {
    return (static_cast<float>(rotary_dim)
            * std::log(static_cast<float>(original_max_position_embeddings)
                       / (static_cast<float>(num_rotations) * 2.0f * kPi)))
         / (2.0f * std::log(base));
}

std::pair<float, float> yarn_find_correction_range(
    int low_rot,
    int high_rot,
    size_t rotary_dim,
    float base,
    size_t original_max_position_embeddings,
    bool truncate) {
    float low = yarn_find_correction_dim(
        low_rot, rotary_dim, base, original_max_position_embeddings);
    float high = yarn_find_correction_dim(
        high_rot, rotary_dim, base, original_max_position_embeddings);
    if (truncate) {
        low = std::floor(low);
        high = std::ceil(high);
    }
    low = std::max(low, 0.0f);
    high = std::min(high, static_cast<float>(rotary_dim) - 1.0f);
    return {low, high};
}

float yarn_get_mscale(float scale, float mscale_coeff = 1.0f) {
    if (scale <= 1.0f) {
        return 1.0f;
    }
    return 0.1f * mscale_coeff * std::log(scale) + 1.0f;
}

} // anonymous namespace

// YarnRopeScalingConfig Implementation
YarnRopeScalingConfig::YarnRopeScalingConfig(
    float factor,
    size_t original_max_position_embeddings,
    size_t rotary_dim,
    float rope_theta,
    int beta_fast,
    int beta_slow,
    float mscale,
    float mscale_all_dim)
    : factor_(factor),
      original_max_position_embeddings_(original_max_position_embeddings) {
    if (factor <= 0.0f) {
        throw std::invalid_argument(
            "YarnRopeScalingConfig factor must be positive, got "
            + std::to_string(factor));
    }
    if (original_max_position_embeddings == 0) {
        throw std::invalid_argument(
            "YarnRopeScalingConfig original_max_position_embeddings must be positive");
    }
    if (rope_theta <= 0.0f) {
        throw std::invalid_argument(
            "YarnRopeScalingConfig rope_theta must be positive, got "
            + std::to_string(rope_theta));
    }
    if (rotary_dim < 2 || rotary_dim % 2 != 0) {
        throw std::invalid_argument(
            "YarnRopeScalingConfig rotary_dim must be a positive even number, got "
            + std::to_string(rotary_dim));
    }

    // vLLM: yarn_find_correction_range(beta_fast, beta_slow, ...)
    auto [low, high] = yarn_find_correction_range(
        beta_fast,
        beta_slow,
        rotary_dim,
        rope_theta,
        original_max_position_embeddings,
        true);
    correction_low_ = low;
    correction_high_ = high;

    magnitude_scale_ = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim);
}

float YarnRopeScalingConfig::yarn_linear_ramp(size_t dim_idx) const {
    float low = correction_low_;
    float high = correction_high_;
    if (low == high) {
        high += 0.001f; // Prevent singularity (matches vLLM)
    }
    float linear = (static_cast<float>(dim_idx) - low) / (high - low);
    return std::clamp(linear, 0.0f, 1.0f);
}

float YarnRopeScalingConfig::get_freq_scale(
    size_t /*pos*/,
    size_t dim_idx,
    float /*base_inv_freq*/) const {
    constexpr float kExtrapolationFactor = 1.0f;
    float ramp = yarn_linear_ramp(dim_idx);
    float inv_freq_mask = (1.0f - ramp) * kExtrapolationFactor;
    return (1.0f - inv_freq_mask) / factor_ + inv_freq_mask;
}

float YarnRopeScalingConfig::get_magnitude_scale(
    size_t /*pos*/,
    size_t /*dim_idx*/,
    float /*base_inv_freq*/) const {
    return magnitude_scale_;
}

} // namespace infinicore::nn
