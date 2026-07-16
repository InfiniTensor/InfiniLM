#pragma once
#include <memory>
#include <vector>

namespace infinicore::nn {

/**
 * @brief Abstract base class for RoPE scaling strategies.
 * Uses polymorphism to eliminate type checking (if-else) in the core RoPE loop.
 */
class RopeScalingConfig {
public:
    virtual ~RopeScalingConfig() = default;

    /**
     * @brief Calculate the frequency scaling factor for a specific position and dimension.
     *
     * @param pos Current sequence position
     * @param dim_idx Current dimension index (0 to head_dim/2 - 1)
     * @param base_inv_freq Pre-computed base inverse frequency for this dimension (1.0 / theta^(2j/head_dim))
     * @return Frequency scaling factor (default 1.0)
     */
    virtual float get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
        return 1.0f;
    }

    /**
     * @brief Calculate the magnitude scaling factor for a specific position and dimension.
     *
     * @param pos Current sequence position
     * @param dim_idx Current dimension index (0 to head_dim/2 - 1)
     * @param base_inv_freq Pre-computed base inverse frequency for this dimension
     * @return Magnitude scaling factor (default 1.0)
     */
    virtual float get_magnitude_scale(size_t pos, size_t dim_idx, float base_inv_freq) const {
        return 1.0f;
    }
};

/**
 * @brief LongRoPE scaling configuration.
 */
class LongRopeScalingConfig : public RopeScalingConfig {
public:
    LongRopeScalingConfig(
        std::vector<float> short_factor,
        std::vector<float> long_factor,
        size_t original_max_position_embeddings,
        float factor = 1.0f);

    float get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const override;
    float get_magnitude_scale(size_t pos, size_t dim_idx, float base_inv_freq) const override;

    size_t original_max_position_embeddings() const { return original_max_position_embeddings_; }
    const std::vector<float> &short_factor() const { return short_factor_; }
    const std::vector<float> &long_factor() const { return long_factor_; }
    float factor() const { return factor_; }

private:
    std::vector<float> short_factor_;
    std::vector<float> long_factor_;
    size_t original_max_position_embeddings_;
    float factor_;
};

// TODO(rubik) implement in cpp
/**
 * @brief Llama3 frequency-aware RoPE scaling configuration.
 * Native support for Llama 3.1 RoPE scaling (smooth interpolation based on wavelength).
 */
class Llama3RopeScalingConfig : public RopeScalingConfig {
public:
    Llama3RopeScalingConfig(
        float factor,
        float low_freq_factor,
        float high_freq_factor,
        size_t original_max_position_embeddings);

    float get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const override;

    // Llama3 does not use magnitude scaling, so it inherits the default get_magnitude_scale() returning 1.0f

private:
    float factor_;
    float low_freq_factor_;
    float high_freq_factor_;
    size_t original_max_position_embeddings_;
};

/**
 * @brief YaRN (Yet another RoPE extensioN) scaling configuration.
 *
 * rope_scaling fields: factor, original_max_position_embeddings, beta_fast, beta_slow,
 *                      mscale, mscale_all_dim
 * Model fields (must match RoPE): rotary_dim (e.g. qk_rope_head_dim), rope_theta
 */
class YarnRopeScalingConfig : public RopeScalingConfig {
public:
    YarnRopeScalingConfig(
        float factor,
        size_t original_max_position_embeddings,
        size_t rotary_dim,
        float rope_theta,
        int beta_fast = 32,
        int beta_slow = 1,
        float mscale = 1.0f,
        float mscale_all_dim = 0.0f);

    float get_freq_scale(size_t pos, size_t dim_idx, float base_inv_freq) const override;
    float get_magnitude_scale(size_t pos, size_t dim_idx, float base_inv_freq) const override;

    /** Recommended RoPE cache length: original_max_position_embeddings * factor. */
    static size_t max_seq_len(float factor, size_t original_max_position_embeddings) {
        return static_cast<size_t>(
            static_cast<float>(original_max_position_embeddings) * factor);
    }

    float factor() const { return factor_; }
    size_t original_max_position_embeddings() const { return original_max_position_embeddings_; }

private:
    float yarn_linear_ramp(size_t dim_idx) const;

    float factor_;
    size_t original_max_position_embeddings_;
    float magnitude_scale_;
    float correction_low_;
    float correction_high_;
};

} // namespace infinicore::nn
