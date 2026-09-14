#pragma once

#include "../qwen3_5/qwen3_5_vision.hpp"
#include "qwen3_vl_text_model.hpp"

namespace infinilm::models::qwen3_vl {

class Qwen3VLModel : public infinicore::nn::Module {
public:
    Qwen3VLModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

protected:
    INFINICORE_NN_MODULE(qwen3_5::Qwen35VisionModel, visual);
    INFINICORE_NN_MODULE(Qwen3VLTextModel, language_model);
};

class Qwen3VLForConditionalGeneration : public InfinilmModel {
public:
    Qwen3VLForConditionalGeneration(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(Qwen3VLModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);

private:
    bool condition_encoder_mode_{false};
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_vl
