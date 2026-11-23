#pragma once

#include "deepseek_v4_decoder_layer.hpp"
#include "deepseek_v4_mtp.hpp"
#include "../../layers/embedding/embedding.hpp"
#include "../infinilm_model.hpp"
#include <infiniccl/infiniccl.h>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4Model final : public infinilm::weights::ModelWeights {
public:
    DeepseekV4Model(infinilm::runtime::Runtime &runtime,
                    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const engine::distributed::RankInfo &rank_info,
                    backends::AttentionBackend attention_backend,
                    const layers::moe::EPConfig &ep_config,
                    bool use_mla,
                    std::shared_ptr<infinilm::layers::RoPE> rope,
                    std::shared_ptr<infinilm::layers::RoPE> compressed_rope);

    infinilm::runtime::Tensor forward(const infinilm::InfinilmModel::Input &input,
                                      infinilm::global_state::ForwardContext &forward_context) const;
    infinilm::runtime::Tensor head_hidden(
        infinilm::runtime::Runtime &runtime,
        const infinilm::runtime::Tensor &target_hidden_states) const;

private:
    std::shared_ptr<infinilm::config::ModelConfig> config_;
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    std::vector<std::shared_ptr<DeepseekV4DecoderLayer>> layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;
    infinilm::weights::Weight hc_head_fn_;
    infinilm::weights::Weight hc_head_base_;
    infinilm::weights::Weight hc_head_scale_;
    infinilm::runtime::Tensor hc_head_rms_weight_;
    size_t hc_mult_{4};
    size_t hc_sinkhorn_iters_{20};
    double hc_eps_{1e-6};
};

class DeepseekV4ForCausalLM final : public infinilm::InfinilmModel {
public:
    DeepseekV4ForCausalLM(infinilm::runtime::Runtime &runtime,
                          std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          const engine::distributed::RankInfo &rank_info,
                          backends::AttentionBackend attention_backend,
                          const layers::moe::EPConfig &ep_config,
                          bool use_mla);

    Output forward(const Input &input, infinilm::global_state::ForwardContext &forward_context) const override;
    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank) override;
    bool supports_kv_only_prepare() const override {
        return true;
    }

private:
    bool use_mla_{true};
    std::shared_ptr<infinilm::layers::RoPE> rope_;
    std::shared_ptr<infinilm::layers::RoPE> compressed_rope_;
    std::shared_ptr<DeepseekV4Model> model_;
    std::shared_ptr<infinilm::layers::linear::ColumnParallelLinear> lm_head_;
    infinilm::runtime::Size tp_rank_{0};
    infinilm::runtime::Size tp_size_{1};
    infinicclComm_t tp_comm_{nullptr};
    std::shared_ptr<DeepseekV4MTP> mtp_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::deepseek_v4
