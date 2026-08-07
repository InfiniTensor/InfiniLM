#include "kimi_k3_text_model.hpp"

#include "kimi_k3_pipeline_partition.hpp"

#include "../../global_state/global_state.hpp"

#include <infinicore/ops/distributed/allgather.hpp>
#include <infinicore/ops/distributed/send_recv.hpp>
#include <infinicore/ops/matmul.hpp>
#include <infinicore/ops/softmax.hpp>

#include <optional>
#include <stdexcept>
#include <tuple>

namespace infinilm::models::kimi_k3 {

KimiK3TextModel::KimiK3TextModel(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device)
    : dtype_(model_config->get_dtype()),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      attn_res_block_size_(model_config->get<size_t>("attn_res_block_size")),
      device_(device) {
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    pp_size_ = static_cast<size_t>(rank_info.pp_size);
    pp_stage_ = static_cast<size_t>(rank_info.pp_stage);
    tp_size_ = static_cast<size_t>(rank_info.tp_size);
    tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
    std::tie(local_layer_begin_, local_layer_end_) = kimi_k3_pipeline_layer_range(num_layers, pp_size_, pp_stage_);
    block_residual_capacity_ = (local_layer_end_ + attn_res_block_size_ - 1) / attn_res_block_size_ + 1;

    if (is_first_pp_stage()) {
        INFINICORE_NN_MODULE_INIT(embed_tokens,
                                  model_config->get<size_t>("vocab_size"), hidden_size_,
                                  std::nullopt, dtype_, device);
    }
    layers_.reserve(local_layer_end_ - local_layer_begin_);
    for (size_t layer_idx = local_layer_begin_; layer_idx < local_layer_end_; ++layer_idx) {
        layers_.push_back(this->register_module<KimiK3DecoderLayer>(
            "layers." + std::to_string(layer_idx), model_config, layer_idx, device));
    }
    if (is_last_pp_stage()) {
        const double eps = model_config->get<double>("rms_norm_eps");
        INFINICORE_NN_MODULE_INIT(output_attn_res_norm, hidden_size_, eps, dtype_, device);
        INFINICORE_NN_MODULE_INIT(output_attn_res_proj,
                                  hidden_size_, 1, false, dtype_, device);
        INFINICORE_NN_MODULE_INIT(norm, hidden_size_, eps, dtype_, device);
    }
}

infinicore::Tensor KimiK3TextModel::embed_tokens(
    const infinicore::Tensor &input_ids) const {
    if (!embed_tokens_) {
        throw std::runtime_error("KimiK3TextModel::embed_tokens called outside the first PP stage");
    }
    return embed_tokens_->forward(input_ids);
}

KimiK3TextModel::PipelineState KimiK3TextModel::initial_state(
    const infinicore::Tensor &first_stage_hidden) const {
    const auto shape = first_stage_hidden->shape();
    auto block_residual_storage = infinicore::Tensor::empty(
        {shape[0] * shape[1], block_residual_capacity_, hidden_size_}, dtype_, device_);
    if (is_first_pp_stage()) {
        return {
            first_stage_hidden,
            block_residual_storage,
            0,
        };
    }
    const size_t prior_block_count = (local_layer_begin_ + attn_res_block_size_ - 1) / attn_res_block_size_;
    auto received_hidden = recv_sharded_last_dim(
        {shape[0], shape[1], hidden_size_});
    auto received_residuals = recv_sharded_last_dim(
        {shape[0] * shape[1], prior_block_count, hidden_size_});
    block_residual_storage->narrow({{1, 0, prior_block_count}})
        ->copy_from(received_residuals);
    return {
        received_hidden,
        block_residual_storage,
        prior_block_count,
    };
}

infinicore::Tensor KimiK3TextModel::recv_sharded_last_dim(
    const infinicore::Shape &shape) const {
    if (hidden_size_ % tp_size_ != 0 || shape.back() != hidden_size_) {
        throw std::runtime_error("KimiK3TextModel: invalid PP receive shape");
    }
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    const size_t shard_hidden = hidden_size_ / tp_size_;
    size_t outer = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        outer *= shape[i];
    }
    auto local = infinicore::op::distributed::recv(
        {outer, shard_hidden},
        dtype_,
        device_,
        static_cast<int>((pp_stage_ - 1) * tp_size_ + tp_rank_),
        rank_info.world_comm);
    if (tp_size_ == 1) {
        return local->view(shape);
    }
    auto gathered = infinicore::op::distributed::allgather(local, tp_size_, rank_info.comm);
    auto restored = gathered->view({tp_size_, outer, shard_hidden})
                        ->permute({1, 0, 2})
                        ->contiguous()
                        ->view({outer, hidden_size_});
    return restored->view(shape);
}

void KimiK3TextModel::send_sharded_last_dim(
    const infinicore::Tensor &tensor) const {
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    const auto shape = tensor->shape();
    if (hidden_size_ % tp_size_ != 0 || shape.back() != hidden_size_) {
        throw std::runtime_error("KimiK3TextModel: invalid PP send shape");
    }
    const size_t shard_hidden = hidden_size_ / tp_size_;
    size_t outer = tensor->numel() / hidden_size_;
    auto local = tensor->view({outer, hidden_size_})
                     ->narrow({{1, tp_rank_ * shard_hidden, shard_hidden}})
                     ->contiguous();
    infinicore::op::distributed::send(
        local,
        static_cast<int>((pp_stage_ + 1) * tp_size_ + tp_rank_),
        rank_info.world_comm);
}

void KimiK3TextModel::send_pipeline_state(const PipelineState &state) const {
    send_sharded_last_dim(state.hidden_states);
    auto block_residual = state.block_residual_storage
                              ->narrow({{1, 0, state.block_residual_count}})
                              ->contiguous();
    send_sharded_last_dim(block_residual);
}

infinicore::Tensor KimiK3TextModel::apply_output_attn_res(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &block_residual_storage,
    size_t block_residual_count) const {
    const auto shape = hidden_states->shape();
    auto hidden_2d = hidden_states->view({shape[0] * shape[1], hidden_size_});
    auto values = block_residual_storage->narrow(
        {{1, 0, block_residual_count + 1}});
    values->narrow({{1, block_residual_count, 1}})
        ->copy_from(hidden_2d->unsqueeze(1));
    auto normalized = output_attn_res_norm_->forward(values);
    auto scores = output_attn_res_proj_->forward(normalized);
    infinicore::op::softmax_(scores, scores, 1);
    auto score_matrix = scores->permute({0, 2, 1})->contiguous();
    return infinicore::op::matmul(score_matrix, values)
        ->squeeze(1)
        ->view(shape);
}

infinicore::Tensor KimiK3TextModel::forward(
    const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto first_hidden = is_first_pp_stage()
                          ? embed_tokens_->forward(input_ids)
                          : infinicore::Tensor::empty(
                              {input_ids->size(0), input_ids->size(1), hidden_size_},
                              dtype_, device_);
    return forward_embeds(first_hidden);
}

infinicore::Tensor KimiK3TextModel::forward_embeds(
    const infinicore::Tensor &inputs_embeds) const {
    auto state = initial_state(inputs_embeds);
    for (const auto &layer : layers_) {
        state.hidden_states = layer->forward(
            state.hidden_states,
            state.block_residual_storage,
            state.block_residual_count);
    }
    if (!is_last_pp_stage()) {
        send_pipeline_state(state);
        return state.hidden_states;
    }
    auto output = apply_output_attn_res(
        state.hidden_states,
        state.block_residual_storage,
        state.block_residual_count);
    return norm_->forward(output);
}

} // namespace infinilm::models::kimi_k3
