#pragma once

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "../../utils/layer_hidden_dump.hpp"
#include "../../config/model_config.hpp"
#include "../../models/infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <vector>

namespace infinilm::layers::causal_lm_templates {

/**
 * @brief Text model architecture (without language modeling head).
 *
 * Generic transformer model consisting of:
 * - Token embeddings
 * - Multiple decoder layers
 * - Final layer normalization
 *
 * @tparam DecoderLayer The decoder layer type (e.g., Qwen3DecoderLayer)
 */
template <typename DecoderLayer>
class TextModel : public infinicore::nn::Module {
public:
    TextModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              const infinicore::Device &device) {
        const auto &dtype{model_config->get_dtype()};
        size_t vocab_size = model_config->get<size_t>("vocab_size");
        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        double rope_theta = model_config->get<double>("rope_theta");
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        embed_tokens_ = this->register_module<infinicore::nn::Embedding>("embed_tokens", vocab_size, hidden_size, std::nullopt, dtype, device);

        layers_.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            layers_.push_back(this->register_module<DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
        }

        norm_ = this->register_module<infinicore::nn::RMSNorm>("norm", hidden_size, rms_norm_eps, dtype, device);
    }

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const {
        auto input_ids = input.input_ids.value();
        auto positions = input.position_ids.value();
        // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
        auto hidden_states = embed_tokens_->forward(input_ids);
        {
            size_t valid_len = 0;
            const auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
            if (piecewise.valid_seq_len > 0) {
                valid_len = piecewise.valid_seq_len;
            }
            if (infinilm::utils::layer_hidden_dump_enabled()
                && infinilm::utils::layer_hidden_dump_all_ranks()
                && infinilm::utils::layer_hidden_dump_this_rank_writes()) {
                infinicore::context::setDevice(hidden_states->device());
                infinicore::context::syncStream();
                auto dump_hidden = infinicore::Tensor::empty(
                    hidden_states->shape(), hidden_states->dtype(), hidden_states->device());
                dump_hidden->copy_from(hidden_states);
                infinicore::context::syncStream();
                infinilm::utils::eager_dump_barrier("eager_dump_embed");
                infinilm::utils::dump_layer_hidden(dump_hidden, 0, valid_len, "embed");
            } else {
                infinilm::utils::eager_dump_barrier("eager_dump_embed");
                infinilm::utils::dump_layer_hidden(hidden_states, 0, valid_len, "embed");
            }
        }

        // 2. Process through all decoder layers
        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                positions,
                hidden_states,
                residual);
            size_t valid_len = 0;
            const auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
            if (piecewise.valid_seq_len > 0) {
                valid_len = piecewise.valid_seq_len;
            }
            infinilm::utils::eager_dump_barrier("eager_dump_layer_full");
            infinilm::utils::dump_layer_hidden(hidden_states, i, valid_len);
        }

        norm_->forward_inplace(hidden_states, residual);
        return hidden_states;
    }

    infinicore::Tensor forward_naive(const infinilm::InfinilmModel::Input &input) const {
        auto input_ids = input.input_ids.value();
        auto positions = input.position_ids.value();
        auto hidden_states = embed_tokens_->forward(input_ids);
        size_t num_layers = layers_.size();
        for (size_t i = 0; i < num_layers; ++i) {
            hidden_states = layers_.at(i)->forward(positions, hidden_states);
        }
        hidden_states = norm_->forward(hidden_states);
        return hidden_states;
    }

    infinicore::Tensor forward_embeds(const infinicore::Tensor &inputs_embeds,
                                      const infinicore::Tensor &position_ids) const {

        auto hidden_states = inputs_embeds;

        //  Process through all decoder layers
        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                position_ids,
                hidden_states,
                residual);
        }

        norm_->forward_inplace(hidden_states, residual);
        return hidden_states;
    }

    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const {
        return embed_tokens_->forward(input_ids);
    }

    size_t num_layers() const { return layers_.size(); }

    void piecewise_embed(const infinilm::InfinilmModel::Input &input,
                         infinicore::Tensor &hidden_states) const {
        auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
        const size_t bucket = hidden_states->size(1);
        const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : bucket;
        auto embedded = embed_tokens_->forward(input.input_ids.value());
        hidden_states->narrow({{1, 0, valid_len}})->copy_from(embedded->narrow({{1, 0, valid_len}}));
        if (valid_len < bucket) {
            auto tail = hidden_states->narrow({{1, valid_len, bucket - valid_len}});
            set_zeros(tail);
        }
    }

    void piecewise_pre_attn_layer(size_t layer_idx,
                                  const infinilm::InfinilmModel::Input &input,
                                  infinicore::Tensor &hidden_states,
                                  infinicore::Tensor &residual) const {
        auto &staging = infinilm::global_state::get_forward_context().piecewise.layer_staging.at(layer_idx);
        layers_.at(layer_idx)->piecewise_pre_attn(
            input.position_ids.value(), hidden_states, residual, staging);
    }

    void piecewise_pre_attn_layernorm_layer(size_t layer_idx,
                                            const infinilm::InfinilmModel::Input &,
                                            infinicore::Tensor &hidden_states,
                                            infinicore::Tensor &residual) const {
        layers_.at(layer_idx)->piecewise_pre_attn_layernorm(hidden_states, residual);
    }

    void piecewise_pre_attn_rope_layer(size_t layer_idx,
                                       const infinilm::InfinilmModel::Input &input,
                                       infinicore::Tensor &,
                                       infinicore::Tensor &) const {
        auto &staging = infinilm::global_state::get_forward_context().piecewise.layer_staging.at(layer_idx);
        layers_.at(layer_idx)->piecewise_pre_attn_rope(input.position_ids.value(), staging);
    }

    void piecewise_pre_attn_staging_layer(size_t layer_idx,
                                          const infinilm::InfinilmModel::Input &,
                                          infinicore::Tensor &hidden_states,
                                          infinicore::Tensor &) const {
        auto &staging = infinilm::global_state::get_forward_context().piecewise.layer_staging.at(layer_idx);
        layers_.at(layer_idx)->piecewise_pre_attn_staging(hidden_states, staging);
    }

    void piecewise_eager_attn_layer(size_t layer_idx,
                                    const infinilm::InfinilmModel::Input &input) const {
        auto &staging = infinilm::global_state::get_forward_context().piecewise.layer_staging.at(layer_idx);
        layers_.at(layer_idx)->piecewise_eager_attn(input.position_ids.value(), staging);
    }

    void piecewise_post_attn_layer(size_t layer_idx,
                                   const infinilm::InfinilmModel::Input &,
                                   infinicore::Tensor &hidden_states,
                                   infinicore::Tensor &residual) const {
        auto &staging = infinilm::global_state::get_forward_context().piecewise.layer_staging.at(layer_idx);
        layers_.at(layer_idx)->piecewise_post_attn(hidden_states, residual, staging);
    }

    void piecewise_post_attn_graph_layer(size_t layer_idx,
                                         const infinilm::InfinilmModel::Input &,
                                         infinicore::Tensor &hidden_states,
                                         infinicore::Tensor &residual) const {
        auto &staging = infinilm::global_state::get_forward_context().piecewise.layer_staging.at(layer_idx);
        layers_.at(layer_idx)->piecewise_post_attn_graph(hidden_states, residual, staging);
    }

    void piecewise_post_attn_mlp_graph_layer(size_t layer_idx,
                                             const infinilm::InfinilmModel::Input &,
                                             infinicore::Tensor &hidden_states,
                                             infinicore::Tensor &residual) const {
        layers_.at(layer_idx)->piecewise_post_attn_mlp_graph(hidden_states, residual);
    }

    void piecewise_post_attn_allreduce_layer(size_t layer_idx,
                                             const infinilm::InfinilmModel::Input &,
                                             infinicore::Tensor &hidden_states,
                                             infinicore::Tensor &residual) const {
        auto &staging = infinilm::global_state::get_forward_context().piecewise.layer_staging.at(layer_idx);
        layers_.at(layer_idx)->piecewise_post_attn_allreduce(hidden_states, residual, staging);
    }

    void piecewise_o_proj_staging_layer(size_t layer_idx,
                                        const infinilm::InfinilmModel::Input &,
                                        infinicore::Tensor &hidden_states,
                                        infinicore::Tensor &residual) const {
        auto &staging = infinilm::global_state::get_forward_context().piecewise.layer_staging.at(layer_idx);
        layers_.at(layer_idx)->piecewise_o_proj_staging(hidden_states, staging);
    }

    infinicore::Tensor piecewise_lm_head(infinicore::Tensor &hidden_states,
                                         infinicore::Tensor &residual) const {
        auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
        const size_t bucket = hidden_states->size(1);
        const size_t valid_len = piecewise.valid_seq_len > 0 ? piecewise.valid_seq_len : bucket;
        if (valid_len < bucket) {
            auto hidden_narrow = hidden_states->narrow({{1, 0, valid_len}});
            auto residual_narrow = residual->narrow({{1, 0, valid_len}});
            norm_->forward_inplace(hidden_narrow, residual_narrow);
        } else {
            norm_->forward_inplace(hidden_states, residual);
        }
        return hidden_states;
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
};

} // namespace infinilm::layers::causal_lm_templates
