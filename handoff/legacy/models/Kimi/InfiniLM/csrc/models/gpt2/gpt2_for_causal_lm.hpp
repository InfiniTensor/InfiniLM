#pragma once

#include "../../backends/attention_backends.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/attention/backends/attention_layer.hpp"
#include "../../layers/moe/ep/ep_config.hpp"
#include "../../config/model_config.hpp"
#include "../../layers/embedding/embedding.hpp"
#include "../../layers/normalization/layer_norm.hpp"
#include "../../weights/weights.hpp"

namespace infinilm::models::gpt2 {

class GPT2Attention : public infinilm::weights::ModelWeights {
public:
    GPT2Attention(infinilm::runtime::Runtime &runtime,
                  std::shared_ptr<infinilm::config::ModelConfig> config,
                  size_t layer_idx, const engine::distributed::RankInfo &rank,
                  backends::AttentionBackend backend, const layers::moe::EPConfig &ep);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                               const infinilm::runtime::Tensor &hidden_states,
                               infinilm::global_state::ForwardContext &forward_context) const;

private:
    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> o_proj_;
    infinilm::weights::Weight o_proj_bias_;
    infinilm::weights::Weight kv_cache_k_scale_;
    infinilm::weights::Weight kv_cache_v_scale_;

    size_t layer_idx_;
    size_t hidden_size_;
    size_t num_heads_;
    size_t num_kv_heads_;
    size_t head_dim_;
    infinilm::backends::AttentionBackend attention_backend_;
};

class GPT2MLP : public infinilm::weights::ModelWeights {
public:
    GPT2MLP(infinilm::runtime::Runtime &runtime,
            std::shared_ptr<infinilm::config::ModelConfig> config,
            const engine::distributed::RankInfo &rank);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                               const infinilm::runtime::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> c_fc_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> c_proj_;
    infinilm::weights::Weight c_proj_bias_;
    std::string activation_;
};

class GPT2Block : public infinilm::weights::ModelWeights {
public:
    GPT2Block(infinilm::runtime::Runtime &runtime,
              std::shared_ptr<infinilm::config::ModelConfig> config,
              size_t layer_idx, const engine::distributed::RankInfo &rank,
              backends::AttentionBackend backend, const layers::moe::EPConfig &ep);

    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &positions,
                               const infinilm::runtime::Tensor &hidden_states,
                               infinilm::global_state::ForwardContext &forward_context) const;

private:
    std::shared_ptr<infinilm::layers::LayerNorm> ln_1_;
    std::shared_ptr<GPT2Attention> attn_;
    std::shared_ptr<infinilm::layers::LayerNorm> ln_2_;
    std::shared_ptr<GPT2MLP> mlp_;
};

class GPT2Model : public infinilm::weights::ModelWeights {
public:
    GPT2Model(infinilm::runtime::Runtime &runtime,
              std::shared_ptr<infinilm::config::ModelConfig> config,
              const engine::distributed::RankInfo &rank,
              backends::AttentionBackend backend, const layers::moe::EPConfig &ep);

    infinilm::runtime::Tensor forward(const infinilm::InfinilmModel::Input &input,
                               infinilm::global_state::ForwardContext &forward_context) const;

private:
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    std::shared_ptr<infinilm::layers::Embedding> embed_positions_;
    std::vector<std::shared_ptr<GPT2Block>> layers_;
    std::shared_ptr<infinilm::layers::LayerNorm> norm_;
};

class GPT2ForCausalLM : public infinilm::InfinilmModel {
public:
    GPT2ForCausalLM(infinilm::runtime::Runtime &runtime,
                    std::shared_ptr<infinilm::config::ModelConfig> config,
                    const engine::distributed::RankInfo &rank,
                    backends::AttentionBackend backend, const layers::moe::EPConfig &ep);

    Output forward(const Input &input,
                   infinilm::global_state::ForwardContext &forward_context) const override;

private:
    std::shared_ptr<GPT2Model> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig>
create_gpt2_model_config(std::shared_ptr<infinilm::config::ModelConfig> config);

} // namespace infinilm::models::gpt2
