#pragma once

#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"
#include "deepseek_v2_decoder_layer.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::deepseek_v2 {

class DeepseekV2Model : public infinicore::nn::Module {
public:
    DeepseekV2Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DeepseekV2DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
};

class DeepseekV2ForCausalLM : public infinilm::InfinilmModel {
public:
    DeepseekV2ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    INFINICORE_NN_MODULE(DeepseekV2Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

/** Implemented in `deepseek_v2_allocate_kv_cache_tensors.cpp`. */
std::vector<infinicore::Tensor> deepseek_v2_allocate_kv_cache_tensors(
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend);

} // namespace infinilm::models::deepseek_v2
