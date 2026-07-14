#pragma once
#include "../../layers/linear/linear.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../infinilm_model.hpp"
#include "glm_attention.hpp"
#include "glm_moe.hpp"
#include "glm_vocab_parallel.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
namespace infinilm::models::glm_moe_dsa {
class GlmDecoder final : public infinicore::nn::Module {
public:
    GlmDecoder(std::shared_ptr<infinilm::config::ModelConfig>, size_t, const infinicore::Device &);
    void forward(const infinicore::Tensor &, infinicore::Tensor &, infinicore::Tensor &) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);
    INFINICORE_NN_MODULE(GlmAttention, self_attn);
    INFINICORE_NN_MODULE(GlmDenseMLP, dense_mlp);
    INFINICORE_NN_MODULE(GlmMoE, moe_mlp);
    bool moe_{false};
};
class GlmModel final : public infinicore::nn::Module {
public:
    GlmModel(std::shared_ptr<infinilm::config::ModelConfig>, const infinicore::Device &);
    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &) const;

private:
    INFINICORE_NN_MODULE(GlmVocabEmbedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(GlmDecoder, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
};
class GlmForCausalLM final : public infinilm::InfinilmModel {
public:
    GlmForCausalLM(std::shared_ptr<infinilm::config::ModelConfig>, const infinicore::Device &);
    Output forward(const Input &) const override;

private:
    INFINICORE_NN_MODULE(GlmModel, model);
    INFINICORE_NN_MODULE(GlmVocabLMHead, lm_head);
};
std::shared_ptr<infinilm::config::ModelConfig> create_glm_config(std::shared_ptr<infinilm::config::ModelConfig>);
} // namespace infinilm::models::glm_moe_dsa
