#pragma once
#include "../../weights/weights.hpp"

#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/embedding/embedding.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/moe/ep/ep_config.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../infinilm_model.hpp"
#include "mamba_state_routing.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::mamba {

class MambaMixer : public infinilm::weights::ModelWeights {
public:
    MambaMixer(infinilm::runtime::Runtime &runtime,
               std::shared_ptr<infinilm::config::ModelConfig> config,
               size_t layer_idx, const engine::distributed::RankInfo &rank_info,
               backends::AttentionBackend attention_backend,
               const layers::moe::EPConfig &ep_config);
    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &hidden_states,
                                      const global_state::ForwardContext &forward_context,
                                      const state_routing::Plan *route = nullptr) const;
    void process_weights_after_loading(infinilm::runtime::Runtime &, bool pre_transpose) override {
        // The model traversal has already processed dt_proj and its other children.
        dt_projection_prepacked_ = pre_transpose;
    }

private:
    size_t layer_idx_;
    size_t intermediate_size_;
    size_t state_size_;
    size_t time_step_rank_;
    size_t conv_kernel_;
    bool dt_projection_prepacked_ = false;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> in_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> x_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> dt_proj_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> out_proj_;
    infinilm::weights::Weight conv1d_weight_;
    infinilm::weights::Weight conv1d_bias_;
    infinilm::weights::Weight A_log_;
    infinilm::weights::Weight D_;
};

class MambaBlock : public infinilm::weights::ModelWeights {
public:
    MambaBlock(infinilm::runtime::Runtime &runtime,
               std::shared_ptr<infinilm::config::ModelConfig> config,
               size_t layer_idx, const engine::distributed::RankInfo &rank_info,
               backends::AttentionBackend attention_backend,
               const layers::moe::EPConfig &ep_config);
    infinilm::runtime::Tensor forward(const infinilm::runtime::Tensor &hidden_states,
                                      const global_state::ForwardContext &forward_context,
                                      const state_routing::Plan *route = nullptr) const;

private:
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;
    std::shared_ptr<MambaMixer> mixer_;
};

class MambaModel : public infinilm::weights::ModelWeights {
public:
    MambaModel(infinilm::runtime::Runtime &runtime,
               std::shared_ptr<infinilm::config::ModelConfig> config,
               const engine::distributed::RankInfo &rank_info,
               backends::AttentionBackend attention_backend,
               const layers::moe::EPConfig &ep_config);
    infinilm::runtime::Tensor forward(const infinilm::InfinilmModel::Input &input,
                                      const global_state::ForwardContext &forward_context) const;

private:
    size_t routing_channels_;
    size_t routing_states_;
    size_t routing_history_;
    infinilm::runtime::DataType routing_dtype_;
    std::shared_ptr<infinilm::layers::Embedding> embeddings_;
    std::vector<std::shared_ptr<MambaBlock>> layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_f_;
};

class MambaForCausalLM : public infinilm::InfinilmModel {
public:
    MambaForCausalLM(infinilm::runtime::Runtime &runtime,
                     std::shared_ptr<infinilm::config::ModelConfig> config,
                     const engine::distributed::RankInfo &rank_info,
                     backends::AttentionBackend attention_backend,
                     const layers::moe::EPConfig &ep_config);
    Output forward(const Input &input, global_state::ForwardContext &forward_context) const override;
    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank_info) override;

private:
    std::shared_ptr<MambaModel> backbone_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig>
create_mamba_model_config(std::shared_ptr<infinilm::config::ModelConfig> config);

} // namespace infinilm::models::mamba
