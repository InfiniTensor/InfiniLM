#pragma once

#include "../../config/model_config.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::ernie4_5_moe_vl {

// Multimodal adapter: VariableResolutionResamplerModel.
// Merges variable-resolution vision patches into the text embedding space:
//   - spatial merge: spatial_conv_size x spatial_conv_size (2x2) neighboring
//     patches are concatenated  -> dim *= 4.
//   - temporal merge (video): temporal_conv_size (2) frames are merged.
//   - projection: -> text hidden_size (2560).
// config: pixel_hidden_size 1280, spatial_conv_size 2, temporal_conv_size 2.
//
// Checkpoint layout under model.resampler_model.* (remapped to visual.merger.*):
//   spatial_linear  : Sequential([0]Linear(5120,5120) -> [1]act -> [2]Linear(5120,5120) -> [3]LayerNorm(5120))
//   temporal_linear : Sequential([0]Linear(10240,5120) -> [1]act -> [2]Linear(5120,5120) -> [3]LayerNorm(5120))
//   mlp             : Linear(5120, 2560)
//   after_norm      : RMSNorm(2560), weight-only (HF uses RMSNorm here, not LayerNorm)
class Ernie4_5_VLResampler : public infinicore::nn::Module {
public:
    Ernie4_5_VLResampler(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);

    // x: vision features [num_patches, pixel_hidden_size].
    // grid_thw: [num_media, 3] = (t, h, w) patch grid, drives the spatial/temporal
    // merge bookkeeping. Returns [num_merged_tokens, text_hidden_size].
    infinicore::Tensor forward(const infinicore::Tensor &x,
                               const infinicore::Tensor &grid_thw) const;

protected:
    size_t pixel_hidden_size_{0};
    size_t text_hidden_size_{0};
    size_t spatial_conv_size_{0};
    size_t temporal_conv_size_{0};

    // spatial_linear Sequential members (registered as "spatial_linear.{0,2,3}")
    std::shared_ptr<infinicore::nn::Linear> spatial_linear_0_;
    std::shared_ptr<infinicore::nn::Linear> spatial_linear_2_;
    std::shared_ptr<infinicore::nn::LayerNorm> spatial_linear_3_;

    // temporal_linear Sequential members (registered as "temporal_linear.{0,2,3}")
    std::shared_ptr<infinicore::nn::Linear> temporal_linear_0_;
    std::shared_ptr<infinicore::nn::Linear> temporal_linear_2_;
    std::shared_ptr<infinicore::nn::LayerNorm> temporal_linear_3_;

    INFINICORE_NN_MODULE(infinicore::nn::Linear, mlp);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, after_norm);
};

} // namespace infinilm::models::ernie4_5_moe_vl
