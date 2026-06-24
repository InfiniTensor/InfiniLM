#pragma once

#include "../../layers/linear/linear.hpp"
#include "../../models/qwen3/qwen3_for_causal_lm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include "qwen3_vl_vision.hpp"

namespace infinilm::models::qwen3_vl {

using Qwen3VLTextModel = infinilm::models::qwen3::Qwen3Model;

class Qwen3VLModel : public infinicore::nn::Module {
public:
    Qwen3VLModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

protected:
    void replace_image_embeddings_(infinicore::Tensor inputs_embeds,
                                   const infinicore::Tensor &input_ids,
                                   const infinicore::Tensor &image_embeds) const;

    size_t image_token_id_;
    INFINICORE_NN_MODULE(Qwen3VLTextModel, language_model);
    INFINICORE_NN_MODULE(Qwen3VLVisionModel, visual);
};

class Qwen3VLForConditionalGeneration : public InfinilmModel {
public:
    Qwen3VLForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                    const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(Qwen3VLModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_vl
