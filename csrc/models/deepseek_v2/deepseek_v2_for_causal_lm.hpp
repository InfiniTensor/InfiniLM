#pragma once
#include "../../weights/weights.hpp"

#include "../../layers/embedding/embedding.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../infinilm_model.hpp"
#include "deepseek_v2_decoder_layer.hpp"
#include "runtime/tensor.hpp"

#include <memory>

namespace infinilm::models::deepseek_v2 {

class DeepseekV2Model : public infinilm::weights::ModelWeights {
public:
    DeepseekV2Model(infinilm::runtime::Runtime &runtime,
                    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const engine::distributed::RankInfo &rank_info,
                    backends::AttentionBackend attention_backend,
                    const layers::moe::EPConfig &ep_config,
                    bool use_mla);

    infinilm::runtime::Tensor forward(const infinilm::InfinilmModel::Input &input,
                                      infinilm::global_state::ForwardContext &forward_context) const;

private:
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    std::vector<std::shared_ptr<DeepseekV2DecoderLayer>> layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;
};

class DeepseekV2ForCausalLM : public infinilm::InfinilmModel {
public:
    DeepseekV2ForCausalLM(infinilm::runtime::Runtime &runtime,
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

private:
    bool use_mla_;
    std::shared_ptr<DeepseekV2Model> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

/** Implemented in `deepseek_v2_allocate_kv_cache_tensors.cpp`. */
std::vector<infinilm::runtime::Tensor> deepseek_v2_allocate_kv_cache_tensors(
    infinilm::runtime::Runtime &runtime,
    const engine::distributed::RankInfo &rank,
    const cache::CacheConfig *cache_config,
    const std::shared_ptr<infinilm::config::ModelConfig> &text_config,
    const backends::AttentionBackend &attention_backend);

} // namespace infinilm::models::deepseek_v2
