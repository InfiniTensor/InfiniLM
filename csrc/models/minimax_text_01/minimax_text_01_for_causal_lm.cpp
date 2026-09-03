#include "minimax_text_01_for_causal_lm.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "minimax_text_01_allocate_kv_cache_tensors.hpp"

#include <infinicore/ops/distributed/allgather.hpp>
#include <infinicore/ops/distributed/send_recv.hpp>
#include <infinicore/ops/select_last_token_hidden.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::minimax_text_01 {

MiniMaxText01Model::MiniMaxText01Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    dtype_ = dtype;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();

    // Pipeline parallel partition: contiguous layer ranges are assigned
    // proportionally so uneven layer counts differ by at most one layer between
    // adjacent PP stages (same scheme as the shared TextModel template).
    pp_size_ = static_cast<size_t>(rank_info.pp_size);
    pp_stage_ = static_cast<size_t>(rank_info.pp_stage);
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
    local_layer_begin_ = num_hidden_layers * pp_stage_ / pp_size_;
    local_layer_end_ = num_hidden_layers * (pp_stage_ + 1) / pp_size_;

    // Only the first pipeline stage owns the embedding table; the last stage
    // owns the final norm (and the LM head lives in the ForCausalLM wrapper).
    if (is_first_pp_stage()) {
        embed_tokens_ = this->register_module<infinicore::nn::Embedding>(
            "embed_tokens", vocab_size, hidden_size_, std::nullopt, dtype, device);
    }
    layers_.reserve(local_layer_end_ - local_layer_begin_);
    for (size_t i = local_layer_begin_; i < local_layer_end_; ++i) {
        layers_.push_back(this->register_module<MiniMaxText01DecoderLayer>(
            "layers." + std::to_string(i), model_config, i, device));
    }
    if (is_last_pp_stage()) {
        norm_ = this->register_module<infinicore::nn::RMSNorm>(
            "norm", hidden_size_, rms_norm_eps, dtype, device);
    }
}

infinicore::Tensor MiniMaxText01Model::forward(const infinilm::InfinilmModel::Input &input) const {
    auto positions = input.position_ids.value();
    auto hidden_states = initial_hidden_states(input);
    for (size_t i = 0; i < layers_.size(); ++i) {
        hidden_states = layers_.at(i)->forward(positions, hidden_states);
    }
    if (!is_last_pp_stage()) {
        // Hand the layer output to the next pipeline stage and bail out: the
        // final norm / LM head live on the last stage only.
        send_pipeline_hidden(hidden_states);
        return hidden_states;
    }
    hidden_states = norm_->forward(hidden_states);
    return hidden_states;
}

infinicore::Tensor MiniMaxText01Model::initial_hidden_states(
    const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    if (is_first_pp_stage()) {
        return embed_tokens_->forward(input_ids);
    }
    auto shape = input_ids->shape();
    return recv_pipeline_hidden(shape[0], shape[1], input_ids->dtype(), input_ids->device());
}

infinicore::Tensor MiniMaxText01Model::recv_pipeline_hidden(
    size_t batch_size,
    size_t seq_len,
    const infinicore::DataType &dtype_hint,
    const infinicore::Device &device) const {
    (void)dtype_hint;
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (hidden_size_ % tp_size_ != 0) {
        throw std::runtime_error("MiniMaxText01Model PP recv requires hidden_size divisible by tp_size");
    }
    const size_t shard_hidden = hidden_size_ / tp_size_;
    // Every TP rank receives the matching hidden-size shard from the same TP
    // rank on the previous stage. A local all-gather then reconstructs the
    // complete hidden state required by the (replicated) decoder input.
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

void MiniMaxText01Model::send_pipeline_hidden(const infinicore::Tensor &hidden_states) const {
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto shape = hidden_states->shape();
    if (shape.size() != 3 || shape[2] != hidden_size_) {
        throw std::runtime_error(
            "MiniMaxText01Model PP send expects hidden states with shape [batch, seq, hidden_size]");
    }
    if (hidden_size_ % tp_size_ != 0) {
        throw std::runtime_error("MiniMaxText01Model PP send requires hidden_size divisible by tp_size");
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

MiniMaxText01ForCausalLM::MiniMaxText01ForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype{model_config->get_dtype()};
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    pp_size_ = static_cast<size_t>(rank_info.pp_size);
    pp_stage_ = static_cast<size_t>(rank_info.pp_stage);

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    if (is_last_pp_stage()) {
        INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
    }
}

infinilm::InfinilmModel::Output MiniMaxText01ForCausalLM::forward(
    const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    if (!is_last_pp_stage()) {
        return {infinicore::Tensor(), hidden_states};
    }

    auto lm_head_input = hidden_states;
    if (!input.sample_all_positions && input.input_offsets.has_value()) {
        const size_t num_requests = input.input_offsets.value()->numel() - 1;
        const bool is_packed_prefill = hidden_states->ndim() == 3
                                    && hidden_states->size(0) == 1
                                    && hidden_states->size(1) > num_requests;
        if (is_packed_prefill) {
            lm_head_input = infinicore::Tensor::empty(
                {1, num_requests, hidden_states->size(2)},
                hidden_states->dtype(),
                hidden_states->device());
            infinicore::op::select_last_token_hidden_(
                lm_head_input, hidden_states, input.input_offsets.value());
        }
    }

    auto logits = lm_head_->forward(lm_head_input);
    return {logits, hidden_states};
}

void MiniMaxText01ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();

    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.kv_cache_vec.clear();
    forward_context.conv_state_vec.clear();
    forward_context.ssm_state_vec.clear();

    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;
    auto cache_vectors = minimax_text_01_allocate_kv_cache_tensors(cache_config, model_config_, attention_backend);
    forward_context.kv_cache_vec = std::move(cache_vectors.kv_cache_tensors);
    forward_context.ssm_state_vec = std::move(cache_vectors.ssm_state_tensors);
}

std::shared_ptr<infinilm::config::ModelConfig> create_minimax_text_01_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minimax_text_01" != model_type) {
        throw std::runtime_error(
            "infinilm::models::minimax_text_01::create_minimax_text_01_model_config: model_type is not minimax_text_01");
    }
    // Bridge MiniMax config field names to the framework MoE expectations.
    model_config->get_config_json()["num_experts"] = model_config->get<size_t>("num_local_experts");
    model_config->get_config_json()["moe_intermediate_size"] = model_config->get<size_t>("intermediate_size");
    return model_config;
}

} // namespace infinilm::models::minimax_text_01

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minimax_text_01,
    infinilm::models::minimax_text_01::MiniMaxText01ForCausalLM,
    infinilm::models::minimax_text_01::create_minimax_text_01_model_config);
} // namespace
