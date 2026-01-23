#pragma once

#include "../infinilm_model.hpp"
#include "../llama/llama_for_causal_lm.hpp"
#include "clip_vision.hpp"
#include "llava_config.hpp"

#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::llava {

class LlavaVisionTower : public infinicore::nn::Module {
public:
    LlavaVisionTower(const ClipVisionConfig &config,
                     const infinicore::DataType &dtype,
                     const infinicore::Device &device);

    infinicore::Tensor forward_features(const infinicore::Tensor &pixel_values,
                                        int64_t feature_layer) const;

private:
    INFINICORE_NN_MODULE(ClipVisionModel, vision_model);
};

class LlavaProjector : public infinicore::nn::Module {
public:
    LlavaProjector(const LlavaConfig &config,
                   const infinicore::DataType &dtype,
                   const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    std::string activation_;
    INFINICORE_NN_MODULE(infinicore::nn::Linear, linear_1);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, linear_2);
};

class LlavaForConditionalGeneration : public InfinilmModel {
public:
    LlavaForConditionalGeneration(const LlavaConfig &config,
                                  const infinicore::Device &device,
                                  engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

    uint32_t compress_kv_cache_inplace(uint32_t seq_len,
                                       size_t batch_size,
                                       const cache::KVCompressionConfig &cfg) override;

private:
    infinicore::Tensor build_merged_embeddings(const infinicore::Tensor &input_ids,
                                               const infinicore::Tensor &inputs_embeds,
                                               const infinicore::Tensor &image_features,
                                               infinicore::Tensor &position_ids_out) const;

    LlavaConfig config_;
    engine::distributed::RankInfo rank_info_;

    INFINICORE_NN_MODULE(llama::LlamaForCausalLM, language_model);
    INFINICORE_NN_MODULE(LlavaVisionTower, vision_tower);
    INFINICORE_NN_MODULE(LlavaProjector, multi_modal_projector);
};

} // namespace infinilm::models::llava
