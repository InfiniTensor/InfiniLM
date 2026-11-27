#pragma once

#include "../../layers/linear/linear.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../../weights/weights.hpp"
#include "gemma4_rope.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <cstddef>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

namespace infinilm::models::gemma4 {

// Gemma4ClippableLinear:
//   Optional input/output clamping via native GPU ReLU identity:
//     max(x, lo) = lo + ReLU(x - lo)
//     min(y, hi) = hi - ReLU(hi - y)
//   Binds exact HF keys: linear.weight, and when use_clipped_linears=true:
//   input_min, input_max, output_min, output_max (reshaped to [1] in loader).
class Gemma4ClippableLinear : public infinilm::weights::ModelWeights {
public:
    Gemma4ClippableLinear(infinilm::runtime::Runtime &runtime,
                          size_t in_features,
                          size_t out_features,
                          bool use_clipped_linears,
                          infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x) const;

private:
    bool use_clipped_linears_{false};
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> linear_;
    infinilm::weights::Weight input_min_;
    infinilm::weights::Weight input_max_;
    infinilm::weights::Weight output_min_;
    infinilm::weights::Weight output_max_;
};

class Gemma4VisionPatchEmbedder : public infinilm::weights::ModelWeights {
public:
    Gemma4VisionPatchEmbedder(infinilm::runtime::Runtime &runtime,
                              const nlohmann::json &vision_cfg,
                              infinilm::runtime::DataType dtype);

    // pixel_values: [num_patches, 3 * patch_size * patch_size] in [0, 1] (or already cast to dtype)
    // pos_x, pos_y: [num_patches] INT32/INT64 coordinates on device
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &pixel_values,
                                      const infinilm::runtime::Tensor &pos_x,
                                      const infinilm::runtime::Tensor &pos_y) const;

private:
    size_t hidden_size_{768};
    size_t patch_size_{16};
    size_t position_embedding_size_{10240};
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> input_proj_;
    infinilm::weights::Weight position_embedding_table_; // [2, position_embedding_size, hidden_size]
};

class Gemma4VisionAttention : public infinilm::weights::ModelWeights {
public:
    Gemma4VisionAttention(infinilm::runtime::Runtime &runtime,
                          const nlohmann::json &vision_cfg,
                          std::shared_ptr<Gemma4RoPE> axial_rope,
                          infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::runtime::Tensor &pos_x,
                                      const infinilm::runtime::Tensor &pos_y) const;

private:
    size_t hidden_size_{768};
    size_t num_heads_{12};
    size_t num_kv_heads_{12};
    size_t head_dim_{64};
    double rms_norm_eps_{1e-6};

    std::shared_ptr<Gemma4ClippableLinear> q_proj_;
    std::shared_ptr<Gemma4ClippableLinear> k_proj_;
    std::shared_ptr<Gemma4ClippableLinear> v_proj_;
    std::shared_ptr<Gemma4ClippableLinear> o_proj_;
    std::shared_ptr<infinilm::layers::RMSNorm> q_norm_;
    std::shared_ptr<infinilm::layers::RMSNorm> k_norm_;
    infinilm::runtime::Tensor v_norm_ones_;
    std::shared_ptr<Gemma4RoPE> axial_rope_;
};

class Gemma4VisionMLP : public infinilm::weights::ModelWeights {
public:
    Gemma4VisionMLP(infinilm::runtime::Runtime &runtime,
                    const nlohmann::json &vision_cfg,
                    infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &x) const;

private:
    std::shared_ptr<Gemma4ClippableLinear> gate_proj_;
    std::shared_ptr<Gemma4ClippableLinear> up_proj_;
    std::shared_ptr<Gemma4ClippableLinear> down_proj_;
};

class Gemma4VisionEncoderLayer : public infinilm::weights::ModelWeights {
public:
    Gemma4VisionEncoderLayer(infinilm::runtime::Runtime &runtime,
                             const nlohmann::json &vision_cfg,
                             std::shared_ptr<Gemma4RoPE> axial_rope,
                             infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      const infinilm::runtime::Tensor &pos_x,
                                      const infinilm::runtime::Tensor &pos_y) const;

private:
    std::shared_ptr<Gemma4VisionAttention> self_attn_;
    std::shared_ptr<Gemma4VisionMLP> mlp_;
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> pre_feedforward_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_feedforward_layernorm_;
};

class Gemma4VisionEncoder : public infinilm::weights::ModelWeights {
public:
    Gemma4VisionEncoder(infinilm::runtime::Runtime &runtime,
                        const nlohmann::json &vision_cfg,
                        infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &inputs_embeds,
                                      const infinilm::runtime::Tensor &pos_x,
                                      const infinilm::runtime::Tensor &pos_y) const;

private:
    std::shared_ptr<Gemma4RoPE> axial_rope_;
    std::vector<std::shared_ptr<Gemma4VisionEncoderLayer>> layers_;
};

class Gemma4VisionModel : public infinilm::weights::ModelWeights {
public:
    Gemma4VisionModel(infinilm::runtime::Runtime &runtime,
                      const nlohmann::json &vision_cfg,
                      infinilm::runtime::DataType dtype);

    // pixel_values: [grid_h * grid_w, 3 * patch_size * patch_size]
    // Returns pooled & (optionally) standardized features [out_tokens, vision_hidden_size] in model dtype.
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &pixel_values,
                                      size_t grid_h,
                                      size_t grid_w) const;

private:
    size_t hidden_size_{768};
    size_t pooling_kernel_size_{3};
    bool standardize_{false};
    infinilm::runtime::DataType dtype_{infinilm::runtime::DataType::kBFloat16};

    std::shared_ptr<Gemma4VisionPatchEmbedder> patch_embedder_;
    std::shared_ptr<Gemma4VisionEncoder> encoder_;
    infinilm::weights::Weight std_bias_;
    infinilm::weights::Weight std_scale_;
};

class Gemma4MultimodalEmbedder : public infinilm::weights::ModelWeights {
public:
    Gemma4MultimodalEmbedder(infinilm::runtime::Runtime &runtime,
                             size_t multimodal_hidden_size,
                             size_t text_hidden_size,
                             double rms_norm_eps,
                             infinilm::runtime::DataType dtype);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                                      const infinilm::runtime::Tensor &inputs_embeds) const;

private:
    double rms_norm_eps_{1e-6};
    infinilm::runtime::Tensor pre_norm_ones_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> embedding_projection_;
};

} // namespace infinilm::models::gemma4
