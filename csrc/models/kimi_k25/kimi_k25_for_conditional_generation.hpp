#pragma once

#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "../../layers/causal_lm_templates/text_model.hpp"
#include "../infinilm_model.hpp"
#include "kimi_k25_decoder_layer.hpp"
#include "kimi_k25_vision.hpp"

#include "../../weights/weights.hpp"

#include <memory>

namespace infinilm::models::kimi_k25 {

using KimiK25TextModel = infinilm::layers::causal_lm_templates::TextModel<KimiK25DecoderLayer>;
using KimiK25LanguageModel = infinilm::layers::causal_lm_templates::TextCausalLM<KimiK25TextModel>;

class KimiK25ForConditionalGeneration : public infinilm::InfinilmModel {
public:
    KimiK25ForConditionalGeneration(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config);

    Output forward(const Input &input,
                   global_state::ForwardContext &forward_context) const override;

protected:
    void replace_image_embeddings(infinilm::runtime::Tensor &inputs_embeds,
                                  const Input &input,
                                  global_state::ForwardContext &forward_context) const;

    std::shared_ptr<KimiK25VisionTower> vision_tower_;
    std::shared_ptr<KimiK25Projector> mm_projector_;
    std::shared_ptr<KimiK25LanguageModel> language_model_;
};

std::shared_ptr<infinilm::config::ModelConfig>
create_kimi_k25_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::kimi_k25
