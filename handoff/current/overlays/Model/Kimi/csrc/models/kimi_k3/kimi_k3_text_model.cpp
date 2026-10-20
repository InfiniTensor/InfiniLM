#include "kimi_k3_trace.hpp"
#include <cstdlib>
#include "kimi_k3_text_model.hpp"
#include "../../execution/submit.hpp"
#include "../models_registry.hpp"
#include <exception>

#include "kimi_k3_pipeline_partition.hpp"

#include "../../global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"

#include "../../engine/distributed/communication_group.hpp"
#include <base/gemm.h>

#include <optional>
#include <stdexcept>
#include <tuple>

namespace infinilm::models::kimi_k3 {

KimiK3TextModel::KimiK3TextModel(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    std::shared_ptr<KimiK3DecoderLayer> pending_child;
    try {
        dtype_ = model_config->get_dtype();
        hidden_size_ = model_config->get<size_t>("hidden_size");
        attn_res_block_size_ = model_config->get<size_t>("attn_res_block_size");
        device_ = runtime.device();
        const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
        pp_size_ = static_cast<size_t>(rank_info.pp_size);
        pp_stage_ = static_cast<size_t>(rank_info.pp_stage);
        tp_size_ = static_cast<size_t>(rank_info.tp_size);
        tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
        tp_comm_ = rank_info.comm;
        world_comm_ = rank_info.world_comm;
        std::tie(local_layer_begin_, local_layer_end_) = kimi_k3_pipeline_layer_range(num_layers, pp_size_, pp_stage_);
        block_residual_capacity_ = (local_layer_end_ + attn_res_block_size_ - 1) / attn_res_block_size_ + 1;

        if (is_first_pp_stage()) {
            embed_tokens_ = std::make_shared<std::remove_reference_t<decltype(*embed_tokens_)>>(
                runtime, model_config->get<size_t>("vocab_size"), hidden_size_, std::nullopt, dtype_);
            this->bind_child("embed_tokens", embed_tokens_);
        }
        layers_.reserve(local_layer_end_ - local_layer_begin_);
        for (size_t layer_idx = local_layer_begin_; layer_idx < local_layer_end_; ++layer_idx) {
            pending_child = std::make_shared<KimiK3DecoderLayer>(
                runtime, model_config, layer_idx, rank_info, attention_backend, ep_config);
            layers_.push_back(pending_child);
            this->bind_child("layers." + std::to_string(layer_idx), pending_child);
        }
        if (is_last_pp_stage()) {
            const double eps = model_config->get<double>("rms_norm_eps");
            output_attn_res_norm_ = std::make_shared<std::remove_reference_t<decltype(*output_attn_res_norm_)>>(
                runtime, hidden_size_, eps, dtype_);
            this->bind_child("output_attn_res_norm", output_attn_res_norm_);
            output_attn_res_proj_ = std::make_shared<std::remove_reference_t<decltype(*output_attn_res_proj_)>>(
                runtime, hidden_size_, 1, false, dtype_);
            this->bind_child("output_attn_res_proj", output_attn_res_proj_);
            norm_ = std::make_shared<std::remove_reference_t<decltype(*norm_)>>(
                runtime, hidden_size_, eps, dtype_);
            this->bind_child("norm", norm_);
        }
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) {
            std::terminate();
        }
        throw;
    }
}

infinilm::runtime::Tensor KimiK3TextModel::embed_tokens(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &input_ids) const {
    if (!embed_tokens_) {
        throw std::runtime_error("KimiK3TextModel::embed_tokens called outside the first PP stage");
    }
    return embed_tokens_->forward(runtime, input_ids);
}

KimiK3TextModel::PipelineState KimiK3TextModel::initial_state(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &first_stage_hidden) const {
    const auto shape = first_stage_hidden->shape();
    auto block_residual_storage = infinilm::runtime::Tensor::empty(
        runtime, {shape[0] * shape[1], block_residual_capacity_, hidden_size_}, dtype_);
    if (is_first_pp_stage()) {
        auto hidden_scratch = block_residual_storage
                                  ->narrow({{1, 0, 1}})
                                  ->view(shape);
        hidden_scratch->copy_from(runtime, first_stage_hidden);
        return {
            hidden_scratch,
            block_residual_storage,
            0,
        };
    }
    const size_t prior_block_count = (local_layer_begin_ + attn_res_block_size_ - 1) / attn_res_block_size_;
    const size_t token_count = shape[0] * shape[1];
    // Transfer all PP state in one message. Keeping state items as the leading
    // dimension leaves the received hidden-state slice contiguous.
    auto received_state = recv_sharded_last_dim(
        runtime,
        {prior_block_count + 1, token_count, hidden_size_});
    block_residual_storage
        ->narrow({{1, 0, prior_block_count + 1}})
        ->copy_from(runtime, received_state->permute({1, 0, 2}));
    auto received_hidden = block_residual_storage
                               ->narrow({{1, prior_block_count, 1}})
                               ->view(shape);
    return {
        received_hidden,
        block_residual_storage,
        prior_block_count,
    };
}

infinilm::runtime::Tensor KimiK3TextModel::recv_sharded_last_dim(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Shape &shape) const {
    if (hidden_size_ % tp_size_ != 0 || shape.back() != hidden_size_) {
        throw std::runtime_error("KimiK3TextModel: invalid PP receive shape");
    }
    const size_t shard_hidden = hidden_size_ / tp_size_;
    size_t outer = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        outer *= shape[i];
    }
    if (runtime.device() != device_) {
        throw std::invalid_argument("Recv runtime and requested devices differ");
    }
    auto local = infinilm::runtime::Tensor::empty(runtime, {outer, shard_hidden}, dtype_);
    infinilm::engine::distributed::recv_(runtime, local, static_cast<int>((pp_stage_ - 1) * tp_size_ + tp_rank_), world_comm_);
    if (tp_size_ == 1) {
        return local->view(shape);
    }
    auto gathered = infinilm::runtime::Tensor::empty(runtime, {outer * tp_size_, shard_hidden}, dtype_);
    infinilm::engine::distributed::allgather_(runtime, gathered, local, tp_comm_);
    auto restored = gathered->view({tp_size_, outer, shard_hidden})
                        ->permute({1, 0, 2})
                        ->contiguous(runtime)
                        ->view({outer, hidden_size_});
    return restored->view(shape);
}

void KimiK3TextModel::send_pipeline_state(infinilm::runtime::Runtime &runtime, const PipelineState &state) const {
    const auto shape = state.hidden_states->shape();
    if (hidden_size_ % tp_size_ != 0 || shape.size() != 3 || shape.back() != hidden_size_) {
        throw std::runtime_error("KimiK3TextModel: invalid PP state shape");
    }
    const size_t shard_hidden = hidden_size_ / tp_size_;
    const size_t token_count = shape[0] * shape[1];
    auto local = infinilm::runtime::Tensor::empty(
        runtime,
        {state.block_residual_count + 1, token_count, shard_hidden},
        dtype_);

    auto state_shard = state.block_residual_storage
                           ->narrow({{1, 0, state.block_residual_count + 1},
                                     {2, tp_rank_ * shard_hidden, shard_hidden}})
                           ->permute({1, 0, 2});
    local->copy_from(runtime, state_shard);

    infinilm::engine::distributed::send(
        runtime,
        local->view({local->numel() / shard_hidden, shard_hidden}),
        static_cast<int>((pp_stage_ + 1) * tp_size_ + tp_rank_),
        world_comm_);
}

infinilm::runtime::Tensor KimiK3TextModel::apply_output_attn_res(
    infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &hidden_states,
    const infinilm::runtime::Tensor &block_residual_storage,
    size_t block_residual_count) const {
    const auto shape = hidden_states->shape();
    auto values = block_residual_storage->narrow(
        {{1, 0, block_residual_count + 1}});
    auto normalized = output_attn_res_norm_->forward(runtime, values);
    auto scores = output_attn_res_proj_->forward(runtime, normalized);
    infinilm::execution::submit<infini::ops::Softmax>(
        runtime, infini::ops::Softmax::DefaultConfig(runtime.device().type()), scores, std::int64_t{1}, std::optional<infini::ops::DataType>{}, scores);
    auto score_matrix = scores->view(
        {scores->size(0), scores->size(2), scores->size(1)});
    auto matmul_output_shape = score_matrix->shape();
    matmul_output_shape.back() = values->size(score_matrix->ndim() - 1);
    auto matmul_output = infinilm::runtime::Tensor::empty(runtime, matmul_output_shape, score_matrix->dtype());
    infinilm::execution::submit<infini::ops::Gemm>(
        runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
        infini::ops::Gemm::DefaultConfig(runtime.device().type()), score_matrix, values,
        std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0f},
        std::optional<float>{0.0f}, std::optional<int>{}, std::optional<int>{}, matmul_output);
    INFINILM_TRACE_OP(Gemm, matmul_output);
    return matmul_output
        ->squeeze(1)
        ->view(shape);
}

infinilm::runtime::Tensor KimiK3TextModel::forward(
    const infinilm::InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    // Eager full-model candidate; recurrent row metadata is supplied by the native harness.
    auto input_ids = input.input_ids.value();
    auto first_hidden = is_first_pp_stage()
                            ? embed_tokens_->forward(runtime, input_ids)
                            : infinilm::runtime::Tensor::empty(
                                  runtime, {input_ids->size(0), input_ids->size(1), hidden_size_},
                                  dtype_);
    return forward_embeds(first_hidden, forward_context);
}

infinilm::runtime::Tensor KimiK3TextModel::forward_embeds(
    const infinilm::runtime::Tensor &inputs_embeds,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    // Eager full-model candidate; recurrent row metadata is supplied by the native harness.
    auto state = initial_state(runtime, inputs_embeds);
    trace_hidden(runtime, state.hidden_states, forward_context, local_layer_begin_, "stage_input");
    size_t trace_layer = local_layer_begin_;
    for (const auto &layer : layers_) {
        state.hidden_states = layer->forward(
            state.hidden_states,
            state.block_residual_storage,
            state.block_residual_count, forward_context);
        trace_hidden(runtime, state.hidden_states, forward_context, trace_layer++, "decoder");
        // Optional correctness diagnosis; performance runs leave both unset.
        const char *device_sync = std::getenv("INFINILM_KIMI_DIAG_DEVICE_SYNC");
        const char *stream_sync = std::getenv("INFINILM_KIMI_DIAG_STREAM_SYNC");
        if (device_sync && device_sync[0] == '1')
            runtime.syncDevice();
        else if (stream_sync && stream_sync[0] == '1')
            runtime.syncStream();
    }
    if (!is_last_pp_stage()) {
        send_pipeline_state(runtime, state);
        return state.hidden_states;
    }
    auto output = apply_output_attn_res(
        runtime,
        state.hidden_states,
        state.block_residual_storage,
        state.block_residual_count);
    trace_hidden(runtime, output, forward_context, 93, "output_attn_res");
    auto final_hidden = norm_->forward(runtime, output);
    trace_hidden(runtime, final_hidden, forward_context, 93, "final_norm");
    return final_hidden;
}

} // namespace infinilm::models::kimi_k3
