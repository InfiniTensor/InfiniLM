#pragma once

#include "ernie4_5_decoder_layer.hpp"
#include "ernie4_5_vision.hpp"

namespace infinilm::models::ernie4_5_moe_vl {

using Ernie4_5Model = infinilm::layers::causal_lm_templates::TextModel<Ernie4_5DecoderLayer>;

class Ernie4_5ForConditionalGeneration : public infinilm::InfinilmModel {
public:
    Ernie4_5ForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device);

    infinilm::InfinilmModel::Output forward(const infinilm::InfinilmModel::Input &input) const override;
    bool supports_mrope_position_ids() const override {
        return true;
    }

private:
    infinicore::Tensor replace_embeddings(const infinicore::Tensor &inputs_embeds,
                                          const infinicore::Tensor &vision_hidden,
                                          const infinicore::Tensor &image_bound) const;

    INFINICORE_NN_MODULE(Ernie4_5Model, model);
    INFINICORE_NN_MODULE(Ernie4_5VisionTransformer, vision_model);
    INFINICORE_NN_MODULE(Ernie4_5VariableResolutionResampler, resampler_model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_ernie4_5_moe_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::ernie4_5_moe_vl
