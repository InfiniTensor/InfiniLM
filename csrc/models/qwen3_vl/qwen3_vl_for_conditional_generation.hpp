#pragma once

#include "../../models/qwen3/qwen3_for_causal_lm.hpp"

namespace infinilm::models::qwen3_vl {

using Qwen3VLTextModel = infinilm::models::qwen3::Qwen3Model;

class Qwen3VLModel : public infinicore::nn::Module {
public:
    Qwen3VLModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

protected:
    std::shared_ptr<Qwen3VLTextModel> language_model_;
};

class Qwen3VLForConditionalGeneration : public InfinilmModel {
public:
    Qwen3VLForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                    const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    std::shared_ptr<Qwen3VLModel> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_vl
