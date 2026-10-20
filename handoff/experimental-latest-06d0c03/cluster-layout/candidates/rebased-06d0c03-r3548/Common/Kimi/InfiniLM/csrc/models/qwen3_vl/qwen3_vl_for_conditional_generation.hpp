#pragma once
#include "../../weights/weights.hpp"

#include "../../models/qwen3/qwen3_for_causal_lm.hpp"

namespace infinilm::models::qwen3_vl {

using Qwen3VLTextModel = infinilm::models::qwen3::Qwen3Model;

class Qwen3VLModel : public infinilm::weights::ModelWeights {
public:
    Qwen3VLModel(infinilm::runtime::Runtime &runtime,
                 std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const engine::distributed::RankInfo &rank,
                 backends::AttentionBackend attention_backend,
                 const layers::moe::EPConfig &ep);

    infinilm::runtime::Tensor forward(const infinilm::InfinilmModel::Input &input,
                                      global_state::ForwardContext &forward_context) const;

protected:
    std::shared_ptr<Qwen3VLTextModel> language_model_;
};

class Qwen3VLForConditionalGeneration : public InfinilmModel {
public:
    Qwen3VLForConditionalGeneration(infinilm::runtime::Runtime &runtime,
                                    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                    const engine::distributed::RankInfo &rank,
                                    backends::AttentionBackend attention_backend,
                                    const layers::moe::EPConfig &ep);

    Output forward(const Input &input, global_state::ForwardContext &forward_context) const override;

    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank) override;

protected:
    std::shared_ptr<Qwen3VLModel> model_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_text_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_vl
