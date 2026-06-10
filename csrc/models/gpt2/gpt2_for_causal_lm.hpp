#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/attention/backends/attention_layer.hpp"
#include "../../config/model_config.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/parameter.hpp"

namespace infinilm::models::gpt2 {

class GPT2Attention : public infinicore::nn::Module {
public:
    GPT2Attention(std::shared_ptr<infinilm::config::ModelConfig> config,
                  size_t layer_idx,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);
    INFINICORE_NN_PARAMETER(o_proj_bias);
    INFINICORE_NN_PARAMETER(kv_cache_k_scale);
    INFINICORE_NN_PARAMETER(kv_cache_v_scale);

    size_t layer_idx_;
    size_t hidden_size_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    infinilm::backends::AttentionBackend attention_backend_;
};

class GPT2MLP : public infinicore::nn::Module {
public:
    GPT2MLP(std::shared_ptr<infinilm::config::ModelConfig> config,
            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, c_fc);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, c_proj);
    INFINICORE_NN_PARAMETER(c_proj_bias);
    std::string activation_;
};

class GPT2Block : public infinicore::nn::Module {
public:
    GPT2Block(std::shared_ptr<infinilm::config::ModelConfig> config,
              size_t layer_idx,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &positions,
                               const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln_1);
    INFINICORE_NN_MODULE(GPT2Attention, attn);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln_2);
    INFINICORE_NN_MODULE(GPT2MLP, mlp);
};

class GPT2Model : public infinicore::nn::Module {
public:
    GPT2Model(std::shared_ptr<infinilm::config::ModelConfig> config,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_positions);
    INFINICORE_NN_MODULE_VEC(GPT2Block, layers);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, norm);
};

class GPT2ForCausalLM : public infinilm::InfinilmModel {
public:
    GPT2ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> config,
                    const infinicore::Device &device);

    Output forward(const Input &input) const override;

private:
    INFINICORE_NN_MODULE(GPT2Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig>
create_gpt2_model_config(std::shared_ptr<infinilm::config::ModelConfig> config);

} // namespace infinilm::models::gpt2
