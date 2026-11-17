#pragma once

#include "../../layers/common_modules.hpp"
#include "../infinilm_model.hpp"
#include "infinicore/tensor.hpp"
#include "qwen3_5_decoderLayer.hpp"
#include "qwen3_5_vision.hpp"

namespace infinilm::models::qwen3_5 {

using Qwen35LanguageModel = infinilm::layers::causal_lm_templates::TextModel<Qwen35DecoderLayer>;

class Qwen35ModelBase : public infinicore::nn::Module {
public:
    Qwen35ModelBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                    const infinicore::Device &device);

    void reset_cache(const cache::CacheConfig *cache_config);

protected:
    void replace_image_embeddings(infinicore::Tensor &inputs_embeds,
                                  const infinilm::InfinilmModel::Input &input) const;

    INFINICORE_NN_MODULE(Qwen35VisionModel, visual);
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

template <typename LanguageModel>
class Qwen35ModelTemplate : public Qwen35ModelBase {
public:
    Qwen35ModelTemplate(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device)
        : Qwen35ModelBase(model_config, device) {
        language_model_ = this->register_module<LanguageModel>(
            "language_model", model_config, device);
    }

    infinicore::Tensor forward(const InfinilmModel::Input &input) const {
        if (input.pixel_values.has_value() && !input.pixel_values->empty()) {
            auto inputs_embeds = language_model_->embed_tokens(input.input_ids.value());
            replace_image_embeddings(inputs_embeds, input);
            return language_model_->forward_embeds(
                inputs_embeds, input.position_ids.value());
        }
        return language_model_->forward(input);
    }

    size_t num_layers() const { return language_model_->num_layers(); }

    void piecewise_embed(const InfinilmModel::Input &input,
                         infinicore::Tensor &hidden_states) const {
        language_model_->piecewise_embed(input, hidden_states);
    }

    void piecewise_pre_attn_layer(size_t layer_idx,
                                  const InfinilmModel::Input &input,
                                  infinicore::Tensor &hidden_states,
                                  infinicore::Tensor &residual) const {
        language_model_->piecewise_pre_attn_layer(layer_idx, input, hidden_states, residual);
    }

    void piecewise_pre_attn_qkv_layer(size_t layer_idx,
                                      const InfinilmModel::Input &input,
                                      infinicore::Tensor &hidden_states,
                                      infinicore::Tensor &residual) const {
        language_model_->piecewise_pre_attn_qkv_layer(layer_idx, input, hidden_states, residual);
    }

    void piecewise_pre_attn_rope_layer(size_t layer_idx,
                                       const InfinilmModel::Input &input) const {
        language_model_->piecewise_pre_attn_rope_layer(layer_idx, input);
    }

    void piecewise_eager_attn_layer(size_t layer_idx,
                                    const InfinilmModel::Input &input) const {
        language_model_->piecewise_eager_attn_layer(layer_idx, input);
    }

    void piecewise_post_attn_layer(size_t layer_idx,
                                   const InfinilmModel::Input &input,
                                   infinicore::Tensor &hidden_states,
                                   infinicore::Tensor &residual) const {
        language_model_->piecewise_post_attn_layer(layer_idx, input, hidden_states, residual);
    }

    void piecewise_post_attn_cg_layer(size_t layer_idx,
                                      const InfinilmModel::Input &input,
                                      infinicore::Tensor &hidden_states,
                                      infinicore::Tensor &residual) const {
        language_model_->piecewise_post_attn_cg_layer(layer_idx, input, hidden_states, residual);
    }

    void piecewise_post_attn_decode_cg_layer(size_t layer_idx,
                                             const InfinilmModel::Input &input,
                                             infinicore::Tensor &hidden_states,
                                             infinicore::Tensor &residual) const {
        language_model_->piecewise_post_attn_decode_cg_layer(
            layer_idx, input, hidden_states, residual);
    }

    void piecewise_eager_moe_layer(size_t layer_idx,
                                   const InfinilmModel::Input &input,
                                   infinicore::Tensor &hidden_states,
                                   infinicore::Tensor &residual) const {
        language_model_->piecewise_eager_moe_layer(layer_idx, input, hidden_states, residual);
    }

    void piecewise_post_attn_graph_layer(size_t layer_idx,
                                         const InfinilmModel::Input &input,
                                         infinicore::Tensor &hidden_states,
                                         infinicore::Tensor &residual) const {
        language_model_->piecewise_post_attn_graph_layer(
            layer_idx, input, hidden_states, residual);
    }

    void piecewise_post_attn_allreduce_layer(size_t layer_idx,
                                             const InfinilmModel::Input &input,
                                             infinicore::Tensor &hidden_states,
                                             infinicore::Tensor &residual) const {
        language_model_->piecewise_post_attn_allreduce_layer(
            layer_idx, input, hidden_states, residual);
    }

    infinicore::Tensor piecewise_lm_head(infinicore::Tensor &hidden_states,
                                         infinicore::Tensor &residual) const {
        return language_model_->piecewise_lm_head(hidden_states, residual);
    }

protected:
    INFINICORE_NN_MODULE(LanguageModel, language_model);
};

using Qwen35Model = Qwen35ModelTemplate<Qwen35LanguageModel>;

} // namespace infinilm::models::qwen3_5
