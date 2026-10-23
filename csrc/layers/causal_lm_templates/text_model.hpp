#include "../../models/models_registry.hpp"
#include "../../weights/weights.hpp"
#include "execution/submit.hpp"
#include <exception>
#pragma once

#include "../../config/inference_memory.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/distributed/communication_group.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../models/infinilm_model.hpp"
#include "../../parallel/topology.hpp"
#include "../embedding/embedding.hpp"
#include "../normalization/rmsnorm.hpp"
#include "execution_capabilities.hpp"
#include "runtime/tensor.hpp"
#include <memory>
#include <stdexcept>
#include <utility>

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
 * @tparam KvOnlyPrepare Opt in only when the decoder has no request state
 *         outside the ordinary ForwardContext KV caches.
 */
template <typename DecoderLayer, bool KvOnlyPrepare = false>
class TextModel : public infinilm::weights::ModelWeights {
public:
    static constexpr bool kv_only_prepare = KvOnlyPrepare;
    static constexpr bool supports_chunked_prefill = KvOnlyPrepare && ChunkedPrefillCapability<DecoderLayer>::value;

    size_t prefill_chunk_size() const { return prefill_chunk_tokens_; }

    TextModel(infinilm::runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> model_config,
              const engine::distributed::RankInfo &rank_info, backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep_config) {
        models::validate_runtime_construction(model_config, runtime, rank_info);
        try {

            const auto &dtype{model_config->get_dtype()};
            dtype_ = dtype;
            size_t vocab_size = model_config->get<size_t>("vocab_size");
            hidden_size_ = model_config->get<size_t>("hidden_size");
            size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
            double rms_norm_eps = model_config->get<double>("rms_norm_eps");

            pp_size_ = static_cast<size_t>(rank_info.pp_size);
            pp_stage_ = static_cast<size_t>(rank_info.pp_stage);
            tp_size_ = static_cast<size_t>(rank_info.tp_size);
            tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
            tp_comm_ = rank_info.comm;
            world_comm_ = rank_info.world_comm;
            if (config::supports_chunked_prefill(*model_config, runtime.device().type(), rank_info.tp_size,
                                                 rank_info.pp_size, attention_backend, supports_chunked_prefill)
                && ep_config.backend == layers::moe::EPBackend::Disabled && ep_config.ep_size == 1) {
                prefill_chunk_tokens_ = config::prefill_chunk_size(*model_config);
            }
            const bool shard_embedding = config::supports_tp_weight_sharding(
                                             *model_config, runtime.device().type(), rank_info.tp_size, rank_info.pp_size)
                                      && hidden_size_ % tp_size_ == 0 && config::tp_embedding_sharding(*model_config);
            // Contiguous layer ranges are assigned proportionally so uneven layer
            // counts differ by at most one layer between adjacent PP stages.
            const auto local_layers = parallel::Topology(tp_size_, rank_info.pp_size,
                                                         rank_info.pp_stage)
                                          .layer_range(num_hidden_layers);
            local_layer_begin_ = local_layers.first;
            local_layer_end_ = local_layers.second;

            if (is_first_pp_stage()) {
                embed_tokens_ = this->bind_child("embed_tokens", std::make_shared<infinilm::layers::Embedding>(runtime, vocab_size, hidden_size_, std::nullopt, dtype,
                                                                                                               shard_embedding ? tp_rank_ : 0, shard_embedding ? tp_size_ : 1, shard_embedding ? tp_comm_ : nullptr));
            }

            layers_.reserve(local_layer_end_ - local_layer_begin_);
            for (size_t i = local_layer_begin_; i < local_layer_end_; ++i) {
                layers_.push_back(this->bind_child("layers." + std::to_string(i), std::make_shared<DecoderLayer>(runtime, model_config, i, rank_info, attention_backend, ep_config)));
            }

            if (is_last_pp_stage()) {
                norm_ = this->bind_child("norm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size_, rms_norm_eps, dtype));
            }
            if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
                runtime.syncStream();
            }
        } catch (...) {
            if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
                try {
                    runtime.syncStream();
                } catch (...) { std::terminate(); }
            }
            throw;
        }
    }

    infinilm::runtime::Tensor forward(
        const infinilm::InfinilmModel::Input &input,
        infinilm::global_state::ForwardContext &forward_context) const {
        if (forward_context.runtime == nullptr) {
            throw std::invalid_argument("text model forward requires an explicit Runtime");
        }
        auto positions = input.position_ids.value();
        if (prefill_chunk_tokens_ && pp_size_ == 1 && !forward_context.runtime->isGraphRecording()
            && !forward_context.nsa_owner && !input.pixel_values && !input.target_hidden_states
            && input.input_ids && input.input_ids.value()->ndim() == 2
            && input.input_ids.value()->size(0) == 1
            && input.input_ids.value()->size(1) > prefill_chunk_tokens_
            && forward_context.attn_metadata.packed_prefill_host
            && forward_context.attn_metadata.block_tables && forward_context.attn_metadata.slot_mapping) {
            return forward_chunked(input, positions, forward_context);
        }
        auto hidden_states = initial_hidden_states(input, *forward_context.runtime);
        return forward_hidden(positions, std::move(hidden_states), forward_context);
    }

    infinilm::runtime::Tensor forward_embeds(
        const infinilm::runtime::Tensor &inputs_embeds,
        const infinilm::runtime::Tensor &position_ids,
        infinilm::global_state::ForwardContext &forward_context) const {
        if (forward_context.runtime == nullptr) {
            throw std::invalid_argument("text model forward_embeds requires an explicit Runtime");
        }
        auto hidden_states = is_first_pp_stage()
                               ? inputs_embeds
                               : recv_pipeline_hidden(inputs_embeds->shape()[0], inputs_embeds->shape()[1],
                                                      inputs_embeds->dtype(), inputs_embeds->device(),
                                                      *forward_context.runtime);
        return forward_hidden(position_ids, std::move(hidden_states), forward_context);
    }

    infinilm::runtime::Tensor embed_tokens(
        const infinilm::runtime::Tensor &input_ids,
        infinilm::global_state::ForwardContext &forward_context) const {
        if (forward_context.runtime == nullptr) {
            throw std::invalid_argument("text model embedding requires an explicit Runtime");
        }
        if (!embed_tokens_) {
            throw std::runtime_error("TextModel::embed_tokens called on a non-first pipeline stage");
        }
        return embed_tokens_->forward(*forward_context.runtime, input_ids);
    }

protected:
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    std::vector<std::shared_ptr<DecoderLayer>> layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;

private:
    infinilm::runtime::Tensor forward_chunked(
        const infinilm::InfinilmModel::Input &input,
        const infinilm::runtime::Tensor &positions,
        infinilm::global_state::ForwardContext &context) const {
        auto &runtime = context.require_runtime();
        const auto &host = *context.attn_metadata.packed_prefill_host;
        const size_t tokens = input.input_ids.value()->size(1);
        struct Chunk {
            size_t start, count;
            infinilm::global_state::ForwardContext context;
        };
        std::vector<Chunk> chunks;
        auto i32 = [&](std::initializer_list<int32_t> values) {
            auto result = infinilm::runtime::Tensor::empty(runtime, {values.size()}, infinilm::runtime::DataType::kInt32);
            runtime.memcpyH2D(result->data(), values.begin(), values.size() * sizeof(int32_t), false);
            return result;
        };
        // Materialize all small control tensors before the first collective.
        for (size_t request = 0; request < host.past.size(); ++request) {
            const size_t query = static_cast<size_t>(host.offsets[request + 1] - host.offsets[request]);
            for (size_t offset = 0; offset < query; offset += prefill_chunk_tokens_) {
                const size_t count = std::min(prefill_chunk_tokens_, query - offset);
                const size_t start = static_cast<size_t>(host.offsets[request]) + offset;
                const int32_t past = host.past[request] + static_cast<int32_t>(offset);
                const int32_t total = past + static_cast<int32_t>(count);
                Chunk chunk{start, count, context};
                auto &metadata = chunk.context.attn_metadata;
                metadata.packed_prefill_host.reset();
                metadata.input_offsets = i32({0, static_cast<int32_t>(count)});
                metadata.cu_seqlens = i32({0, total});
                metadata.past_sequence_lengths = i32({past});
                metadata.total_sequence_lengths = i32({total});
                metadata.block_tables = context.attn_metadata.block_tables.value()->narrow({{0, request, 1}});
                metadata.slot_mapping = context.attn_metadata.slot_mapping.value()->narrow({{0, start, count}});
                metadata.max_query_length = count;
                metadata.max_sequence_length = total;
                metadata.logical_kv_upper_bound = total;
                metadata.first_past_sequence_length = past;
                metadata.first_total_sequence_length = total;
                chunks.push_back(std::move(chunk));
            }
        }
        auto output = infinilm::runtime::Tensor::empty(runtime, {1, tokens, hidden_size_}, dtype_);
        auto initial = initial_hidden_states(input, runtime);
        for (auto &chunk : chunks) {
            auto chunk_positions = positions->narrow({{positions->ndim() - 1, chunk.start, chunk.count}});
            auto hidden = initial->narrow({{1, chunk.start, chunk.count}});
            auto result = forward_hidden(chunk_positions, std::move(hidden), chunk.context);
            auto destination = output->narrow({{1, chunk.start, chunk.count}});
            runtime.memcpyD2D(destination->data(), result->data(), result->nbytes(), true);
        }
        return output;
    }

    infinilm::runtime::Tensor forward_hidden(
        const infinilm::runtime::Tensor &positions,
        infinilm::runtime::Tensor hidden_states,
        infinilm::global_state::ForwardContext &forward_context) const {
        infinilm::runtime::Tensor residual;
        // PP sends an already-rounded sum; retain the interior layer's
        // residual-present normalization path at the receiving stage.
        if (!is_first_pp_stage() && !layers_.empty()) {
            residual = infinilm::runtime::Tensor::empty(
                *forward_context.runtime, hidden_states->shape(), hidden_states->dtype());
            infinilm::execution::submit<infini::ops::Fill>(
                *forward_context.runtime,
                infini::ops::Fill::DefaultConfig(forward_context.runtime->device().type()),
                residual, 0.0, residual);
        }
        for (size_t i = 0; i < layers_.size(); ++i) {
            layers_.at(i)->forward(
                positions, hidden_states, residual, forward_context);
        }

        if (!is_last_pp_stage()) {
            hidden_states = materialize_hidden_states(
                hidden_states, residual, *forward_context.runtime);
            send_pipeline_hidden(hidden_states, *forward_context.runtime);
            return hidden_states;
        }

        norm_->forward_inplace(*forward_context.runtime, hidden_states, residual);
        return hidden_states;
    }

    bool is_first_pp_stage() const { return pp_stage_ == 0; }
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

    infinilm::runtime::Tensor initial_hidden_states(
        const infinilm::InfinilmModel::Input &input,
        infinilm::runtime::Runtime &runtime) const {
        auto input_ids = input.input_ids.value();
        if (is_first_pp_stage()) {
            return embed_tokens_->forward(runtime, input_ids);
        }
        auto shape = input_ids->shape();
        return recv_pipeline_hidden(shape[0], shape[1], input_ids->dtype(), input_ids->device(), runtime);
    }

    infinilm::runtime::Tensor recv_pipeline_hidden(size_t batch_size,
                                                   size_t seq_len,
                                                   const infinilm::runtime::DataType &dtype_hint,
                                                   const infinilm::runtime::Device &device,
                                                   infinilm::runtime::Runtime &runtime) const {
        (void)dtype_hint;
        if (hidden_size_ % tp_size_ != 0) {
            throw std::runtime_error("TextModel PP recv requires hidden_size divisible by tp_size");
        }
        const size_t shard_hidden = hidden_size_ / tp_size_;
        // Every TP rank receives the matching hidden-size shard from the same
        // TP rank on the previous node. The local all-gather reconstructs the
        // complete hidden state required by the replicated decoder input.
        const auto shape = infinilm::runtime::Shape{batch_size * seq_len, shard_hidden};
        const int peer = static_cast<int>((pp_stage_ - 1) * tp_size_ + tp_rank_);
        infinilm::runtime::Tensor gathered;
        if (tp_size_ > 1) {
            // Allocators may wait on their stream. Do not allocate between a
            // pending cross-stage Recv and its dependent local AllGather.
            const auto gathered_shape = infinilm::runtime::Shape{
                tp_size_ * batch_size * seq_len, shard_hidden};
            gathered = infinilm::runtime::Tensor::empty(runtime, gathered_shape, dtype_);
        }
        if (runtime.device() != device) {
            throw std::invalid_argument("Recv runtime and requested devices differ");
        }
        auto local_shard = infinilm::runtime::Tensor::empty(runtime, shape, dtype_);
        infinilm::engine::distributed::recv_(runtime, local_shard, peer, world_comm_);
        if (tp_size_ == 1) {
            return local_shard->view({batch_size, seq_len, hidden_size_});
        }
        infinilm::engine::distributed::allgather_(
            runtime, gathered, local_shard, tp_comm_);
        auto reordered = gathered->view({tp_size_, batch_size, seq_len, shard_hidden})
                             ->permute({1, 2, 0, 3});
        return reordered->contiguous(runtime)
            ->view({batch_size, seq_len, hidden_size_});
    }

    void send_pipeline_hidden(const infinilm::runtime::Tensor &hidden_states,
                              infinilm::runtime::Runtime &runtime) const {
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
        auto shard_view = hidden_states->narrow({{2, tp_rank_ * shard_hidden, shard_hidden}});
        auto local_shard = shard_view->contiguous(runtime)
                               ->view({shape[0] * shape[1], shard_hidden});
        const int peer = static_cast<int>((pp_stage_ + 1) * tp_size_ + tp_rank_);
        infinilm::engine::distributed::send(runtime, local_shard, peer, world_comm_);
    }

    infinilm::runtime::Tensor materialize_hidden_states(infinilm::runtime::Tensor &hidden_states,
                                                        infinilm::runtime::Tensor &residual,
                                                        infinilm::runtime::Runtime &runtime) const {
        if (!residual) {
            return hidden_states;
        }
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, residual->shape(), residual->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), residual, hidden_states, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        return ops_output;
    }

    infinilm::runtime::DataType dtype_{infinilm::runtime::DataType::kFloat32};
    size_t hidden_size_{0};
    size_t prefill_chunk_tokens_{0};
    size_t pp_size_{1};
    size_t pp_stage_{0};
    size_t tp_size_{1};
    size_t tp_rank_{0};
    infinicclComm_t tp_comm_{nullptr};
    infinicclComm_t world_comm_{nullptr};
    size_t local_layer_begin_{0};
    size_t local_layer_end_{0};
};

} // namespace infinilm::layers::causal_lm_templates
