#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace infinicore::nn {
class RoPE;
}

namespace infinilm::models::deepseek_v4 {

// Per-layer RoPE for DeepSeek-V4 attention: main rope (sliding) or compress rope (CSA/HCA + YaRN).
// forward() uses infinicore::nn::RoPE on GPU when available; block/inverse paths stay on CPU reference.
class DeepseekV4RoPE {
public:
    enum class Backend {
        CPU,
        GPU,
        Auto, // GPU when device != CPU and numerics-compatible; else CPU reference
    };

    DeepseekV4RoPE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   size_t layer_idx,
                   const infinicore::Device &device,
                   Backend backend = Backend::Auto);

    const DeepseekV4RopeParams &params() const { return params_; }
    size_t compress_ratio() const { return compress_ratio_; }

    // Partial interleaved forward RoPE on x [B, S, H, D] (trailing rope_dim slice).
    infinicore::Tensor forward(const infinicore::Tensor &x,
                               const std::vector<int64_t> &positions) const;

    // Block-level RoPE on compressed KV entries stored in a flat float buffer [B, nb, D].
    void forward_blocks(std::vector<float> &kv_comp,
                        size_t batch_size,
                        size_t nb,
                        size_t head_dim,
                        size_t seq_len,
                        const std::vector<int64_t> &positions) const;

    // Inverse RoPE on a single head vector inside a flat buffer (attention output path).
    void inverse_at_offset(std::vector<float> &values,
                           size_t offset,
                           int64_t position) const;

private:
    bool use_gpu_forward_() const;
    const std::shared_ptr<infinicore::nn::RoPE> &active_gpu_rope_() const;
    infinicore::Tensor forward_gpu_(const infinicore::Tensor &x,
                                    const std::vector<int64_t> &positions) const;
    infinicore::Tensor forward_cpu_(const infinicore::Tensor &x,
                                      const std::vector<int64_t> &positions) const;

    DeepseekV4RopeParams params_;
    size_t compress_ratio_{0};
    Backend backend_{Backend::Auto};
    infinicore::Device device_;
    // One instance per layer: sliding layers use main_rope_; compress layers use compress_rope_ (YaRN).
    std::shared_ptr<infinicore::nn::RoPE> main_rope_;
    std::shared_ptr<infinicore::nn::RoPE> compress_rope_;
};

} // namespace infinilm::models::deepseek_v4
