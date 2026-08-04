#pragma once

#include "../../config/model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allgather.hpp"
#include "infinicore/ops/distributed/send_recv.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <stdexcept>

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
        dtype_ = dtype;
        size_t vocab_size = model_config->get<size_t>("vocab_size");
        hidden_size_ = model_config->get<size_t>("hidden_size");
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();

        pp_size_ = static_cast<size_t>(rank_info.pp_size);
        pp_stage_ = static_cast<size_t>(rank_info.pp_stage);
        tp_size_ = static_cast<size_t>(rank_info.tp_size);
        tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
        // Contiguous layer ranges are assigned proportionally so uneven layer
        // counts differ by at most one layer between adjacent PP stages.
        local_layer_begin_ = num_hidden_layers * pp_stage_ / pp_size_;
        local_layer_end_ = num_hidden_layers * (pp_stage_ + 1) / pp_size_;

        if (is_first_pp_stage()) {
            embed_tokens_ = this->register_module<infinicore::nn::Embedding>("embed_tokens", vocab_size, hidden_size_, std::nullopt, dtype, device);
        }

        layers_.reserve(local_layer_end_ - local_layer_begin_);
        for (size_t i = local_layer_begin_; i < local_layer_end_; ++i) {
            layers_.push_back(this->register_module<DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
        }

        if (is_last_pp_stage()) {
            norm_ = this->register_module<infinicore::nn::RMSNorm>("norm", hidden_size_, rms_norm_eps, dtype, device);
        }
    }

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const {
        auto positions = input.position_ids.value();
        auto hidden_states = initial_hidden_states(input);

        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                positions,
                hidden_states,
                residual);
        }

        if (!is_last_pp_stage()) {
            hidden_states = materialize_hidden_states(hidden_states, residual);
            send_pipeline_hidden(hidden_states);
            return hidden_states;
        }

        norm_->forward_inplace(hidden_states, residual);
        return hidden_states;
    }

    infinicore::Tensor forward_naive(const infinilm::InfinilmModel::Input &input) const {
        auto positions = input.position_ids.value();
        auto hidden_states = initial_hidden_states(input);
        size_t num_layers = layers_.size();
        for (size_t i = 0; i < num_layers; ++i) {
            hidden_states = layers_.at(i)->forward(positions, hidden_states);
        }
        if (!is_last_pp_stage()) {
            send_pipeline_hidden(hidden_states);
            return hidden_states;
        }
        hidden_states = norm_->forward(hidden_states);
        return hidden_states;
    }

    infinicore::Tensor forward_embeds(const infinicore::Tensor &inputs_embeds,
                                      const infinicore::Tensor &position_ids) const {

        auto hidden_states = is_first_pp_stage() ? inputs_embeds : recv_pipeline_hidden(inputs_embeds->shape()[0], inputs_embeds->shape()[1], inputs_embeds->dtype(), inputs_embeds->device());

        //  Process through all decoder layers
        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                position_ids,
                hidden_states,
                residual);
        }

        if (!is_last_pp_stage()) {
            hidden_states = materialize_hidden_states(hidden_states, residual);
            send_pipeline_hidden(hidden_states);
            return hidden_states;
        }

        norm_->forward_inplace(hidden_states, residual);
        return hidden_states;
    }

    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const {
        if (!embed_tokens_) {
            throw std::runtime_error("TextModel::embed_tokens called on a non-first pipeline stage");
        }
        return embed_tokens_->forward(input_ids);
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

private:
    bool is_first_pp_stage() const { return pp_stage_ == 0; }
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

    infinicore::Tensor initial_hidden_states(const infinilm::InfinilmModel::Input &input) const {
        auto input_ids = input.input_ids.value();
        if (is_first_pp_stage()) {
            return embed_tokens_->forward(input_ids);
        }
        auto shape = input_ids->shape();
        return recv_pipeline_hidden(shape[0], shape[1], input_ids->dtype(), input_ids->device());
    }

    infinicore::Tensor recv_pipeline_hidden(size_t batch_size,
                                            size_t seq_len,
                                            const infinicore::DataType &dtype_hint,
                                            const infinicore::Device &device) const {
        (void)dtype_hint;
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
        if (hidden_size_ % tp_size_ != 0) {
            throw std::runtime_error("TextModel PP recv requires hidden_size divisible by tp_size");
        }
        const size_t shard_hidden = hidden_size_ / tp_size_;
        // Every TP rank receives the matching hidden-size shard from the same
        // TP rank on the previous node. The local all-gather reconstructs the
        // complete hidden state required by the replicated decoder input.
        auto local_shard = infinicore::op::distributed::recv(
            {batch_size * seq_len, shard_hidden},
            dtype_,
            device,
            static_cast<int>((pp_stage_ - 1) * tp_size_ + tp_rank_),
            rank_info.world_comm);
        if (tp_size_ == 1) {
            return local_shard->view({batch_size, seq_len, hidden_size_});
        }
        auto gathered = infinicore::op::distributed::allgather(local_shard, tp_size_, rank_info.comm);
        return gathered->view({tp_size_, batch_size, seq_len, shard_hidden})
            ->permute({1, 2, 0, 3})
            ->contiguous()
            ->view({batch_size, seq_len, hidden_size_});
    }

    void send_pipeline_hidden(const infinicore::Tensor &hidden_states) const {
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
        auto shape = hidden_states->shape();
        if (shape.size() != 3 || shape[2] != hidden_size_) {
            throw std::runtime_error("TextModel PP send expects hidden states with shape [batch, seq, hidden_size]");
        }
        if (hidden_size_ % tp_size_ != 0) {
            throw std::runtime_error("TextModel PP send requires hidden_size divisible by tp_size");
        }
        const size_t shard_hidden = hidden_size_ / tp_size_;
        // Split hidden_size across local TP ranks. Matching ranks transfer their
        // shards independently, avoiding one full activation transfer per rank.
        auto local_shard = hidden_states->narrow({{2, tp_rank_ * shard_hidden, shard_hidden}})
                               ->contiguous()
                               ->view({shape[0] * shape[1], shard_hidden});
        infinicore::op::distributed::send(
            local_shard,
            static_cast<int>((pp_stage_ + 1) * tp_size_ + tp_rank_),
            rank_info.world_comm);
    }

    infinicore::Tensor materialize_hidden_states(infinicore::Tensor &hidden_states,
                                                 infinicore::Tensor &residual) const {
        if (!residual) {
            return hidden_states;
        }
        return infinicore::op::add(residual, hidden_states);
    }

    infinicore::DataType dtype_{infinicore::DataType::F32};
    size_t hidden_size_{0};
    size_t pp_size_{1};
    size_t pp_stage_{0};
    size_t tp_size_{1};
    size_t tp_rank_{0};
    size_t local_layer_begin_{0};
    size_t local_layer_end_{0};
};

} // namespace infinilm::layers::causal_lm_templates
