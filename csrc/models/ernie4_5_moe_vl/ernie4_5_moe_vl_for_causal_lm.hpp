#pragma once

#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"
#include "ernie4_5_decoder_layer.hpp"
#include "ernie4_5_resampler.hpp"
#include "ernie4_5_vision.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"

#include <memory>

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5Model : public infinicore::nn::Module {
public:
    Ernie4_5Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;
    infinicore::Tensor forward_embeds(const infinicore::Tensor &inputs_embeds,
                                      const infinicore::Tensor &position_ids,
                                      const infinicore::Tensor &token_type_ids = infinicore::Tensor()) const;
    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const;
    infinicore::Tensor resample_vision(const infinicore::Tensor &vision_features,
                                       const infinicore::Tensor &grid_thw) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(Ernie4_5DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(Ernie4_5Resampler, resampler_model);
};

class Ernie4_5MoeVLForCausalLM : public InfinilmModel {
public:
    Ernie4_5MoeVLForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device);

    Output forward(const Input &input) const override;
    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const;
    Ernie4_5Model &model() { return *model_; }

private:
    infinicore::Tensor build_multimodal_embeds_(const Input &input) const;
    infinicore::Tensor replace_image_embeds_(const infinicore::Tensor &input_ids,
                                             const infinicore::Tensor &inputs_embeds,
                                             const infinicore::Tensor &image_features) const;
    infinicore::Tensor concat_optional_tensors_(const std::optional<std::vector<infinicore::Tensor>> &tensors,
                                                int dim) const;

    size_t im_patch_id_{100295};

    INFINICORE_NN_MODULE(Ernie4_5VisionModel, vision_model);
    INFINICORE_NN_MODULE(Ernie4_5Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig>
create_ernie4_5_moe_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::ernie4_5_moe_vl
