#pragma once
#include "../../weights/weights.hpp"

#include "../../layers/common_modules.hpp"
#include "../../layers/embedding/embedding.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "../infinilm_model.hpp"
#include "ernie4_5_decoder_layer.hpp"
#include "ernie4_5_vision.hpp"

#include <vector>

namespace infinilm::models::ernie4_5_vl {

class Ernie45Model : public infinilm::weights::ModelWeights {
public:
    Ernie45Model(infinilm::runtime::Runtime &runtime,
                 std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const engine::distributed::RankInfo &rank_info,
                 backends::AttentionBackend attention_backend,
                 const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(const InfinilmModel::Input &input,
                                      global_state::ForwardContext &forward_context,
                                      const Ernie45VisionModel *vision_model = nullptr) const;
    infinilm::runtime::Tensor forward_embeds(infinilm::runtime::Tensor hidden_states,
                                             const infinilm::runtime::Tensor &positions,
                                             global_state::ForwardContext &forward_context) const;

protected:
    std::shared_ptr<Ernie45ResamplerModel> resampler_model_;
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    std::vector<std::shared_ptr<Ernie45DecoderLayer>> layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;

private:
    void replace_embeddings(infinilm::runtime::Runtime &runtime,
                            infinilm::runtime::Tensor inputs_embeds,
                            const infinilm::runtime::Tensor &vision_hidden,
                            const infinilm::runtime::Tensor &image_bound) const;
    void apply_image_embeddings(infinilm::runtime::Tensor inputs_embeds,
                                const InfinilmModel::Input &input,
                                const Ernie45VisionModel &vision_model,
                                global_state::ForwardContext &forward_context) const;

    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    std::shared_ptr<const Ernie45MropeCache> mrope_cache_;
};

} // namespace infinilm::models::ernie4_5_vl
