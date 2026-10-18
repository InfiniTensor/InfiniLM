#pragma once

#include "../../config/model_config.hpp"
#include "../infinilm_model.hpp"
#include "../qwen2/qwen2_for_causal_lm.hpp"
#include "resampler.hpp"
#include "siglip_vision.hpp"

#include "../../weights/weights.hpp"
#include "runtime/tensor.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace infinilm::models::minicpmv {

class MiniCPMVModel : public InfinilmModel {
public:
    MiniCPMVModel(infinilm::runtime::Runtime &runtime,
                  std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const engine::distributed::RankInfo &rank,
                  backends::AttentionBackend attention_backend,
                  const layers::moe::EPConfig &ep);

    // Explicit visual path shares this Runtime with patch, encoder and resampler.
    // Data-dependent visual metadata is rejected before recording submissions.
    Output forward(const Input &input, global_state::ForwardContext &forward_context) const override;

    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank) override;

private:
    void replace_embeddings(infinilm::runtime::Tensor inputs_embeds,
                            const infinilm::runtime::Tensor &vision_hidden,
                            const std::vector<int64_t> &image_bounds,
                            infinilm::runtime::Runtime &runtime) const;

    std::shared_ptr<infinilm::config::ModelConfig> config_;
    std::optional<backends::AttentionBackend> attention_backend_;

    std::shared_ptr<infinilm::models::qwen2::Qwen2ForCausalLM> llm_;
    std::shared_ptr<SiglipVisionModel> vpm_;
    std::shared_ptr<Resampler> resampler_;
};

std::shared_ptr<infinilm::config::ModelConfig> create_minicpmv_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::minicpmv
