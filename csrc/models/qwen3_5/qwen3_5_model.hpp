#pragma once
#include "../../weights/weights.hpp"

#include "../../layers/common_modules.hpp"
#include "../infinilm_model.hpp"
#include "qwen3_5_decoderLayer.hpp"
#include "qwen3_5_vision.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <exception>

namespace infinilm::models::qwen3_5 {

using Qwen35LanguageModel = infinilm::layers::causal_lm_templates::TextModel<Qwen35DecoderLayer>;

class Qwen35ModelBase : public infinilm::weights::ModelWeights {
public:
    Qwen35ModelBase(infinilm::runtime::Runtime &runtime,
                    std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const engine::distributed::RankInfo &rank_info,
                    backends::AttentionBackend attention_backend,
                    const layers::moe::EPConfig &ep_config);

    void reset_cache(const cache::CacheConfig *cache_config,
                     global_state::ForwardContext &forward_context,
                     backends::AttentionBackend attention_backend,
                     const engine::distributed::RankInfo &rank_info);

protected:
    void replace_image_embeddings(infinilm::runtime::Tensor &inputs_embeds,
                                  const infinilm::InfinilmModel::Input &input,
                                  global_state::ForwardContext &forward_context) const;

    std::shared_ptr<Qwen35VisionModel> visual_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

template <typename LanguageModel>
class Qwen35ModelTemplate : public Qwen35ModelBase {
public:
    Qwen35ModelTemplate(
        infinilm::runtime::Runtime &runtime,
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config)
        : Qwen35ModelBase(runtime, model_config, rank_info, attention_backend, ep_config) {
        try {
            language_model_ = this->bind_child("language_model", std::make_shared<LanguageModel>(runtime, model_config, rank_info, attention_backend, ep_config));
        } catch (...) {
            // Base visual owners remain alive until this body drain completes.
            try {
                runtime.syncStream();
            } catch (...) { std::terminate(); }
            throw;
        }
    }

    infinilm::runtime::Tensor forward(
        const InfinilmModel::Input &input,
        infinilm::global_state::ForwardContext &forward_context) const {
        auto &runtime = forward_context.require_runtime();
        runtime.assertOwnerThread();
        if (input.pixel_values.has_value() && !input.pixel_values->empty()) {
            if (runtime.isGraphRecording()) {
                throw std::invalid_argument("Qwen35Model: dynamic visual metadata is not recording-qualified");
            }
            auto inputs_embeds = language_model_->embed_tokens(
                input.input_ids.value(), forward_context);
            replace_image_embeddings(inputs_embeds, input, forward_context);
            return language_model_->forward_embeds(
                inputs_embeds, input.position_ids.value(), forward_context);
        }
        return language_model_->forward(input, forward_context);
    }

protected:
    std::shared_ptr<LanguageModel> language_model_;
};

using Qwen35Model = Qwen35ModelTemplate<Qwen35LanguageModel>;

} // namespace infinilm::models::qwen3_5
