#pragma once

#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/parameter.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <optional>
#include <vector>

namespace infinilm::models::rwkv {

class Rwkv5SelfAttention : public infinicore::nn::Module {
public:
    Rwkv5SelfAttention(std::shared_ptr<infinilm::config::ModelConfig> config,
                       size_t layer_idx,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               infinicore::Tensor &attn_x_state,
                               infinicore::Tensor &wkv_state) const;

private:
    infinicore::Tensor shifted_hidden_(const infinicore::Tensor &hidden_states,
                                       infinicore::Tensor &state) const;
    infinicore::Tensor group_norm_(const infinicore::Tensor &x) const;

    size_t layer_idx_;
    size_t hidden_size_;
    size_t attention_hidden_size_;
    size_t head_size_;
    size_t num_heads_;
    size_t head_size_divisor_;

    INFINICORE_NN_PARAMETER(time_decay);
    INFINICORE_NN_PARAMETER(time_faaaa);
    INFINICORE_NN_PARAMETER(time_mix_gate);
    INFINICORE_NN_PARAMETER(time_mix_key);
    INFINICORE_NN_PARAMETER(time_mix_value);
    INFINICORE_NN_PARAMETER(time_mix_receptance);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, key);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, value);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, receptance);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, gate);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, output);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln_x);
};

class Rwkv5FeedForward : public infinicore::nn::Module {
public:
    Rwkv5FeedForward(std::shared_ptr<infinilm::config::ModelConfig> config,
                     size_t layer_idx,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               infinicore::Tensor &ffn_x_state) const;

private:
    infinicore::Tensor shifted_hidden_(const infinicore::Tensor &hidden_states,
                                       infinicore::Tensor &state) const;

    size_t layer_idx_;
    size_t hidden_size_;
    size_t intermediate_size_;

    INFINICORE_NN_PARAMETER(time_mix_key);
    INFINICORE_NN_PARAMETER(time_mix_receptance);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, key);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, receptance);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, value);
};

class Rwkv5Block : public infinicore::nn::Module {
public:
    Rwkv5Block(std::shared_ptr<infinilm::config::ModelConfig> config,
               size_t layer_idx,
               const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               infinicore::Tensor &attn_x_state,
                               infinicore::Tensor &wkv_state,
                               infinicore::Tensor &ffn_x_state) const;

private:
    size_t layer_idx_;
    size_t rescale_every_;
    std::shared_ptr<infinicore::nn::LayerNorm> pre_ln_;
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln1);
    INFINICORE_NN_MODULE(Rwkv5SelfAttention, attention);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln2);
    INFINICORE_NN_MODULE(Rwkv5FeedForward, feed_forward);
};

class Rwkv5Model : public infinicore::nn::Module {
public:
    Rwkv5Model(std::shared_ptr<infinilm::config::ModelConfig> config,
               const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input,
                               infinicore::Tensor &attn_x_state,
                               infinicore::Tensor &wkv_state,
                               infinicore::Tensor &ffn_x_state) const;

private:
    size_t rescale_every_;

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embeddings);
    INFINICORE_NN_MODULE_VEC(Rwkv5Block, blocks);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln_out);
};

class Rwkv5ForCausalLM : public infinilm::InfinilmModel {
public:
    Rwkv5ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> config,
                     const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    void ensure_state_(size_t batch_size) const;

    size_t num_hidden_layers_;
    size_t hidden_size_;
    size_t num_heads_;
    size_t head_size_;
    infinicore::Device device_;
    infinicore::DataType dtype_;
    mutable size_t state_batch_size_ = 0;
    mutable infinicore::Tensor attn_x_state_;
    mutable infinicore::Tensor wkv_state_;
    mutable infinicore::Tensor ffn_x_state_;

    INFINICORE_NN_MODULE(Rwkv5Model, rwkv);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_rwkv5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::rwkv
