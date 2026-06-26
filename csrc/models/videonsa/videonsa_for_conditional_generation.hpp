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
    VideoNSAForConditionalGeneration(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device);

    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

protected:
    void replace_embeddings(infinicore::Tensor inputs_embeds,
                            const infinicore::Tensor &vision_hidden,
                            const infinicore::Tensor &image_bound) const;

    INFINICORE_NN_MODULE(VideoNSATextModel, model);
    INFINICORE_NN_MODULE(VideoNSAVisionModel, visual);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig> create_videonsa_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::videonsa
