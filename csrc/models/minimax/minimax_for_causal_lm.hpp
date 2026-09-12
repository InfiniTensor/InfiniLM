#pragma once

#include "../../cache/mamba_cache.hpp"
#include "../infinilm_model.hpp"
#include "minimax_decoderLayer.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::minimax {

class MiniMaxModel : public infinicore::nn::Module {
public:
    MiniMaxModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    std::vector<std::shared_ptr<MiniMaxDecoderLayer>> layers_;
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
};

class MiniMaxForCausalLM : public InfinilmModel {
public:
    MiniMaxForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    INFINICORE_NN_MODULE(MiniMaxModel, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_minimax_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minimax
