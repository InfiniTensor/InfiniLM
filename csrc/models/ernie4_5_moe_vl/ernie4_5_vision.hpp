#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5VisionPatchEmbed : public infinicore::nn::Module {
public:
    Ernie4_5VisionPatchEmbed(const nlohmann::json &vision_config,
                             const infinicore::DataType &dtype,
                             const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t patch_size_{14};
    size_t in_channels_{3};
    size_t embed_dim_{1280};

    INFINICORE_NN_MODULE(infinilm::nn::Linear, proj);
};

class Ernie4_5VisionAttention : public infinicore::nn::Module {
public:
    Ernie4_5VisionAttention(const nlohmann::json &vision_config,
                            const infinicore::DataType &dtype,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &grid_thw) const;

private:
    infinicore::Tensor apply_rotary_pos_emb_(const infinicore::Tensor &tensor,
                                             const infinicore::Tensor &grid_thw) const;
    infinicore::Tensor segmented_attention_(const infinicore::Tensor &q,
                                            const infinicore::Tensor &k,
                                            const infinicore::Tensor &v,
                                            const infinicore::Tensor &grid_thw) const;

    size_t embed_dim_{1280};
    size_t num_heads_{16};
    size_t head_dim_{80};
    size_t spatial_merge_size_{2};
    float scale_{1.0f};

    INFINICORE_NN_MODULE(infinilm::nn::Linear, qkv);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, proj);
};

class Ernie4_5VisionMLP : public infinicore::nn::Module {
public:
    Ernie4_5VisionMLP(const nlohmann::json &vision_config,
                      const infinicore::DataType &dtype,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    std::string hidden_act_;

    INFINICORE_NN_MODULE(infinilm::nn::Linear, fc1);
    INFINICORE_NN_MODULE(infinilm::nn::Linear, fc2);
};

class Ernie4_5VisionBlock : public infinicore::nn::Module {
public:
    Ernie4_5VisionBlock(const nlohmann::json &vision_config,
                        const infinicore::DataType &dtype,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &grid_thw) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm1);
    INFINICORE_NN_MODULE(Ernie4_5VisionAttention, attn);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm2);
    INFINICORE_NN_MODULE(Ernie4_5VisionMLP, mlp);
};

class Ernie4_5VisionModel : public infinicore::nn::Module {
public:
    Ernie4_5VisionModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &images,
                               const infinicore::Tensor &grid_thw) const;

private:
    infinicore::Tensor normalize_images_(const infinicore::Tensor &images) const;
    infinicore::Tensor vision_grid_thw_(const infinicore::Tensor &grid_thw) const;

    nlohmann::json vision_config_;
    infinicore::DataType dtype_{infinicore::DataType::BF16};
    size_t depth_{32};
    size_t hidden_size_{1280};
    size_t patch_size_{14};

    INFINICORE_NN_MODULE(Ernie4_5VisionPatchEmbed, patch_embed);
    INFINICORE_NN_MODULE_VEC(Ernie4_5VisionBlock, blocks);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln);
};

} // namespace infinilm::models::ernie4_5_moe_vl
