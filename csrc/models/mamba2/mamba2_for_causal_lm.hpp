#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "infinicore/nn/embedding.hpp"
#include "mamba2_mixer.hpp"

namespace infinilm::models::mamba2 {

class Mamba2Model : public infinicore::nn::Module {
public:
    Mamba2Model(std::shared_ptr<config::ModelConfig> config, const infinicore::Device &device);
    infinicore::Tensor forward(const InfinilmModel::Input &input) const;
    infinicore::Tensor embedding_weight() const { return embeddings_->weight(); }

private:
    infinicore::DataType dtype_;
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embeddings);
    INFINICORE_NN_MODULE_VEC(Mamba2Block, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm_f);
};

class Mamba2ForCausalLM : public layers::causal_lm_templates::TextCausalLM<Mamba2Model> {
public:
    Mamba2ForCausalLM(std::shared_ptr<config::ModelConfig> config, const infinicore::Device &device);
    void reset_cache(const cache::CacheConfig *config) override;
};

std::shared_ptr<config::ModelConfig> create_mamba2_model_config(std::shared_ptr<config::ModelConfig> config);

} // namespace infinilm::models::mamba2
