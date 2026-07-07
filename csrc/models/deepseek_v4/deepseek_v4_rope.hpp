#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>

namespace infinicore::nn {
class RoPE;
}

namespace infinilm::models::deepseek_v4 {

// Per-layer GPU RoPE for DeepSeek-V4 attention: main rope (sliding) or compress rope (CSA/HCA + YaRN).
class DeepseekV4RoPE {
public:
    DeepseekV4RoPE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   size_t layer_idx,
                   const infinicore::Device &device);

    const DeepseekV4RopeParams &params() const { return params_; }
    size_t compress_ratio() const { return compress_ratio_; }

    // Apply partial interleaved forward RoPE to Q and K [B, S, H, D] on GPU.
    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &q,
                                                                 const infinicore::Tensor &k,
                                                                 const infinicore::Tensor &pos_ids) const;

private:
    const std::shared_ptr<infinicore::nn::RoPE> &active_gpu_rope_() const;
    infinicore::Tensor prepare_gpu_pos_tensor_(const infinicore::Tensor &pos_ids,
                                               const infinicore::Device &device) const;
    infinicore::Tensor forward_gpu_(const infinicore::Tensor &x,
                                    const infinicore::Tensor &pos_ids,
                                    const std::shared_ptr<infinicore::nn::RoPE> &gpu_rope) const;

    DeepseekV4RopeParams params_;
    size_t compress_ratio_{0};
    infinicore::Device device_;
    std::shared_ptr<infinicore::nn::RoPE> main_rope_;
    std::shared_ptr<infinicore::nn::RoPE> compress_rope_;
};

} // namespace infinilm::models::deepseek_v4
