#pragma once

#include "../../config/model_config.hpp"
#include "../infinilm_model.hpp"
#include "../qwen2/qwen2_for_causal_lm.hpp"
#include "resampler.hpp"
#include "siglip_vision.hpp"

#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::models::minicpmv {

class MiniCPMVModel : public InfinilmModel {
public:
    MiniCPMVModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);

    Output forward(const Input &input) const override;

    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    void replace_embeddings(infinicore::Tensor inputs_embeds,
                            const infinicore::Tensor &vision_hidden,
                            const infinicore::Tensor &image_bound) const;

    std::shared_ptr<infinilm::config::ModelConfig> config_;

    INFINICORE_NN_MODULE(infinilm::models::qwen2::Qwen2ForCausalLM, llm);
    INFINICORE_NN_MODULE(SiglipVisionModel, vpm);
    INFINICORE_NN_MODULE(Resampler, resampler);
};

std::shared_ptr<infinilm::config::ModelConfig> create_minicpmv_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpmv
