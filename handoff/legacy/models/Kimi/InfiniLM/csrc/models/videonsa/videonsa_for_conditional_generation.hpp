#pragma once

#include "../../layers/common_modules.hpp"
#include "videonsa_attention.hpp"
#include "videonsa_vision.hpp"

namespace infinilm::models::videonsa {

using VideoNSAMLP = infinilm::layers::MLP;
using VideoNSADecoderLayer = infinilm::layers::causal_lm_templates::TextDecoderLayer<VideoNSAAttention, VideoNSAMLP>;
using VideoNSATextModel = infinilm::layers::causal_lm_templates::TextModel<VideoNSADecoderLayer>;

class VideoNSAForConditionalGeneration : public InfinilmModel {
public:
    VideoNSAForConditionalGeneration(infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep);

    Output forward(const Input &input, global_state::ForwardContext &forward_context) const override;
    void prepare_paged_owner(global_state::ForwardContext &) const override;
    void finish_paged_owner(bool) const override;
    void invalidate_paged_owner() const noexcept override;
    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank) override;

protected:
    void replace_embeddings(infinilm::runtime::Runtime &runtime,
                            infinilm::runtime::Tensor inputs_embeds,
                            const infinilm::runtime::Tensor &vision_hidden,
                            const infinilm::runtime::Tensor &image_bound) const;

    std::shared_ptr<VideoNSATextModel> model_;
    std::shared_ptr<VideoNSAVisionModel> visual_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_videonsa_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::videonsa
