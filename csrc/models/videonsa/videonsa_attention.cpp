#include "videonsa_attention.hpp"
#include "../../execution/submit.hpp"
#include "../../execution/paged_cache_checks.hpp"
#include <base/reshape_and_cache_flash.h>
#include <base/gemm.h>
#include "../../execution/tensor_expressions.hpp"
#include <exception>
#include "runtime/runtime.hpp"
#include "runtime/utils.hpp"
#include <limits>
#include <stdexcept>
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "runtime/context.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::videonsa {

namespace {

constexpr size_t kNsaBlockSize = 64;
constexpr int kNsaSelectBlocks = 4;

// Source vectors remain alive through completion or fail-stop in this local upload.
void upload_custom_rope_cache(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor table, const infinilm::runtime::Tensor &host) {
    try {
        table->copy_from(runtime, host);
        runtime.syncStream(); // cache creation only; never forward
    } catch (...) {
        const auto failure = std::current_exception();
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try { runtime.syncStream(); } catch (...) { std::terminate(); }
        }
        std::rethrow_exception(failure);
    }
}

// Temporarily creating cache here. Scheduled to be replaced by a revamped mrope module.
std::pair<infinilm::runtime::Tensor, infinilm::runtime::Tensor> build_mrope_cache(infinilm::runtime::Runtime &runtime, size_t max_seq_len,
                                                                    size_t rotary_dim,
                                                                    double theta,
                                                                    const infinilm::runtime::DataType &dtype,
                                                                    const infinilm::runtime::Device &device) {
    runtime.assertOwnerThread();
    if (runtime.device() != device) throw std::invalid_argument("VideoNSA MRoPE cache Runtime and device must agree");
    if (rotary_dim == 0 || rotary_dim % 2 != 0) {
        throw std::invalid_argument("VideoNSAAttention: invalid mrope rotary_dim");
    }
    const size_t cache_dim = rotary_dim / 2;
    const size_t numel = max_seq_len * cache_dim;
    std::vector<float> sin_data(numel);
    std::vector<float> cos_data(numel);
    for (size_t pos = 0; pos < max_seq_len; ++pos) {
        for (size_t dim_idx = 0; dim_idx < cache_dim; ++dim_idx) {
            const float inv_freq = 1.0f / std::pow(static_cast<float>(theta), 2.0f * static_cast<float>(dim_idx) / static_cast<float>(rotary_dim));
            const float angle = static_cast<float>(pos) * inv_freq;
            const size_t offset = pos * cache_dim + dim_idx;
            sin_data[offset] = std::sin(angle);
            cos_data[offset] = std::cos(angle);
        }
    }

    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    auto sin_cache = infinilm::runtime::Tensor::empty(runtime, {max_seq_len, cache_dim}, dtype);
    auto cos_cache = infinilm::runtime::Tensor::empty(runtime, {max_seq_len, cache_dim}, dtype);
    if (dtype == infinilm::runtime::DataType::kFloat32) {
        auto sin_cpu = infinilm::runtime::Tensor::from_blob(sin_data.data(), {max_seq_len, cache_dim}, infinilm::runtime::DataType::kFloat32, cpu);
        auto cos_cpu = infinilm::runtime::Tensor::from_blob(cos_data.data(), {max_seq_len, cache_dim}, infinilm::runtime::DataType::kFloat32, cpu);
        upload_custom_rope_cache(runtime, sin_cache, sin_cpu);
        upload_custom_rope_cache(runtime, cos_cache, cos_cpu);
        return {sin_cache, cos_cache};
    }
    if (dtype == infinilm::runtime::DataType::kBFloat16) {
        std::vector<uint16_t> sin_bf16(numel);
        std::vector<uint16_t> cos_bf16(numel);
        for (size_t i = 0; i < numel; ++i) {
            sin_bf16[i] = f32_to_bf16(sin_data[i]);
            cos_bf16[i] = f32_to_bf16(cos_data[i]);
        }
        auto sin_cpu = infinilm::runtime::Tensor::from_blob(sin_bf16.data(), {max_seq_len, cache_dim}, infinilm::runtime::DataType::kBFloat16, cpu);
        auto cos_cpu = infinilm::runtime::Tensor::from_blob(cos_bf16.data(), {max_seq_len, cache_dim}, infinilm::runtime::DataType::kBFloat16, cpu);
        upload_custom_rope_cache(runtime, sin_cache, sin_cpu);
        upload_custom_rope_cache(runtime, cos_cache, cos_cpu);
        return {sin_cache, cos_cache};
    }
    if (dtype == infinilm::runtime::DataType::kFloat16) {
        std::vector<uint16_t> sin_f16(numel);
        std::vector<uint16_t> cos_f16(numel);
        for (size_t i = 0; i < numel; ++i) {
            sin_f16[i] = f32_to_f16(sin_data[i]);
            cos_f16[i] = f32_to_f16(cos_data[i]);
        }
        auto sin_cpu = infinilm::runtime::Tensor::from_blob(sin_f16.data(), {max_seq_len, cache_dim}, infinilm::runtime::DataType::kFloat16, cpu);
        auto cos_cpu = infinilm::runtime::Tensor::from_blob(cos_f16.data(), {max_seq_len, cache_dim}, infinilm::runtime::DataType::kFloat16, cpu);
        upload_custom_rope_cache(runtime, sin_cache, sin_cpu);
        upload_custom_rope_cache(runtime, cos_cache, cos_cpu);
        return {sin_cache, cos_cache};
    }
    throw std::runtime_error("VideoNSAAttention: mrope cache dtype is unsupported");
}

std::pair<infinilm::runtime::Tensor, infinilm::runtime::Tensor> apply_mrope(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &q,
                                                              const infinilm::runtime::Tensor &k,
                                                              const infinilm::runtime::Tensor &positions,
                                                              const infinilm::runtime::Tensor &cos_cache,
                                                              const infinilm::runtime::Tensor &sin_cache,
                                                              size_t head_dim,
                                                              size_t rotary_dim,
                                                              const std::array<int, 3> &section,
                                                              bool interleaved) {
    const size_t num_tokens = q->size(0);
    auto q_flat = q->contiguous(runtime)->view({num_tokens, q->size(1) * head_dim});
    auto k_flat = k->contiguous(runtime)->view({num_tokens, k->size(1) * head_dim});
    auto q_out = infinilm::runtime::Tensor::empty(runtime, q_flat->shape(), q_flat->dtype());
    auto k_out = infinilm::runtime::Tensor::empty(runtime, k_flat->shape(), k_flat->dtype());
    execution::submit<infini::ops::Mrope>(
        runtime, execution::RecordingPolicy::HostOnly,
        infini::ops::Mrope::DefaultConfig(runtime.device().type()),
        q_flat, k_flat, cos_cache, sin_cache, positions,
        static_cast<int>(head_dim), static_cast<int>(rotary_dim),
        section[0], section[1], section[2], interleaved, q_out, k_out);
    INFINILM_TRACE_OP(MRoPE, q_out);
    return {q_out->view(q->shape()), k_out->view(k->shape())};
}

infinilm::runtime::Tensor mean_pool_blocks(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &x, size_t block_size) {
    const size_t seq_len = x->size(0);
    std::vector<infinilm::runtime::Tensor> blocks;
    blocks.reserve((seq_len + block_size - 1) / block_size);
    for (size_t start = 0; start < seq_len; start += block_size) {
        const size_t len = std::min(block_size, seq_len - start);
        auto block = x->narrow({{0, start, len}});
        const auto pooling_axes = infinilm::execution::tensor_checks::sum::normalized_dim(block, {0});
        auto pooled = infinilm::runtime::Tensor::empty(runtime, infinilm::execution::tensor_checks::sum::reduced_shape(block, pooling_axes, false), block->dtype());
        infinilm::execution::tensor_checks::sum::validate(runtime, pooled, block, pooling_axes, false);
        infinilm::execution::submit<infini::ops::Sum>(runtime, infinilm::execution::RecordingPolicy::DeviceOnly, infini::ops::Sum::DefaultConfig(runtime.device().type()), block, std::optional<std::vector<int64_t>>{std::vector<int64_t>(pooling_axes.begin(), pooling_axes.end())}, false, std::optional<infini::ops::DataType>{}, pooled);
        blocks.push_back(pooled->unsqueeze(0));
    }
    if (blocks.size() == 1) return blocks.front();
    return infinilm::execution::cat_alloc(runtime, blocks, 0);
}

infinilm::runtime::Tensor repeat_group_tensor(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &x, size_t repeats) {
    std::vector<infinilm::runtime::Tensor> copies;
    copies.reserve(repeats);
    for (size_t i = 0; i < repeats; ++i) {
        copies.push_back(x);
    }
    if (copies.size() == 1) return copies.front();
    return infinilm::execution::cat_alloc(runtime, copies, 0);
}

infinilm::runtime::Tensor grouped_dense_attention(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &q,
                                           const infinilm::runtime::Tensor &k,
                                           const infinilm::runtime::Tensor &v,
                                           size_t num_heads,
                                           size_t num_kv_heads,
                                           size_t head_dim,
                                           float scale) {
    const size_t seq_len = q->size(0);
    const size_t kv_len = k->size(0);
    const size_t heads_per_group = num_heads / num_kv_heads;
    std::vector<infinilm::runtime::Tensor> group_outputs;
    group_outputs.reserve(num_kv_heads);
    for (size_t g = 0; g < num_kv_heads; ++g) {
        auto q_group = q->narrow({{1, g * heads_per_group, heads_per_group}})
                           ->permute({1, 0, 2})
                           ->contiguous(runtime);                               // [heads_per_group, seq, dim]
        auto k_group = k->narrow({{1, g, 1}})->squeeze(1)->unsqueeze(0); // [1, kv, dim]
        auto v_group = v->narrow({{1, g, 1}})->squeeze(1)->unsqueeze(0); // [1, kv, dim]
        auto k_repeated = repeat_group_tensor(runtime, k_group, heads_per_group);
        auto v_repeated = repeat_group_tensor(runtime, v_group, heads_per_group);
        const auto scores_rhs = k_repeated->permute({0, 2, 1});
        auto scores_shape = q_group->shape();
        scores_shape[q_group->ndim() - 1] = scores_rhs->size(q_group->ndim() - 1);
        auto scores = infinilm::runtime::Tensor::empty(runtime, scores_shape, q_group->dtype());
        infinilm::execution::submit<infini::ops::Gemm>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
            infini::ops::Gemm::DefaultConfig(runtime.device().type()), q_group, scores_rhs,
            std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{scale},
            std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, scores);
        INFINILM_TRACE_OP(Gemm, scores);
        INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
            && runtime.device().type() != infinilm::runtime::Device::Type::kMars
            && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
        infinilm::execution::submit<infini::ops::Softmax>(
            runtime, infini::ops::Softmax::DefaultConfig(runtime.device().type()), scores, std::int64_t{-1}, std::optional<infini::ops::DataType>{}, scores);
        auto out_shape = scores->shape();
        out_shape[scores->ndim() - 1] = v_repeated->size(scores->ndim() - 1);
        auto out = infinilm::runtime::Tensor::empty(runtime, out_shape, scores->dtype());
        infinilm::execution::submit<infini::ops::Gemm>(
            runtime, infinilm::execution::RecordingPolicy::DeviceOnly,
            infini::ops::Gemm::DefaultConfig(runtime.device().type()), scores, v_repeated,
            std::optional<infinilm::runtime::Tensor>{}, std::optional<float>{1.0F},
            std::optional<float>{0.0F}, std::optional<int>{}, std::optional<int>{}, out);
        INFINILM_TRACE_OP(Gemm, out);
        out = out
                       ->view({heads_per_group, seq_len, head_dim})
                       ->permute({1, 0, 2})
                       ->contiguous(runtime);
        group_outputs.push_back(out);
    }
    if (group_outputs.size() == 1) return group_outputs.front();
    return infinilm::execution::cat_alloc(runtime, group_outputs, 1);
}

infinilm::runtime::Tensor expand_head_gate(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &gate, size_t head_dim) {
    const size_t seq_len = gate->size(1);
    const size_t num_heads = gate->size(2);
    auto flat_gate = gate->contiguous(runtime)->view({seq_len * num_heads});
    std::vector<int64_t> expand_indices(seq_len * num_heads * head_dim);
    for (size_t row = 0; row < seq_len * num_heads; ++row) {
        for (size_t d = 0; d < head_dim; ++d) {
            expand_indices[row * head_dim + d] = static_cast<int64_t>(row);
        }
    }
    auto indices = infinilm::runtime::Tensor::empty(runtime, {seq_len, num_heads, head_dim}, infinilm::runtime::DataType::kInt64);
    auto host = infinilm::runtime::Tensor::from_blob(expand_indices.data(), indices->shape(), indices->dtype(),
                                            infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    indices->copy_from(runtime, host); // host vector is consumed before returning
    auto expanded_gate = infinilm::runtime::Tensor::empty(runtime, indices->shape(), flat_gate->dtype());
    infinilm::execution::tensor_checks::take::validate(runtime, expanded_gate, flat_gate, indices);
    infinilm::execution::submit<infini::ops::Take>(runtime, infinilm::execution::RecordingPolicy::DeviceOnly, infini::ops::Take::DefaultConfig(runtime.device().type()), flat_gate, indices, expanded_gate);
    return expanded_gate;
}

std::array<int, 3> read_mrope_section(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    const auto &config_json = model_config->get_config_json();
    if (!config_json.contains("rope_scaling") || !config_json["rope_scaling"].contains("mrope_section")) {
        throw std::runtime_error("VideoNSAAttention: rope_scaling.mrope_section is required");
    }
    auto section = config_json["rope_scaling"]["mrope_section"].get<std::vector<int>>();
    if (section.size() != 3) {
        throw std::runtime_error("VideoNSAAttention: mrope_section must contain three entries");
    }
    return {section[0], section[1], section[2]};
}

infinilm::runtime::Tensor local_head_gates(const infinilm::runtime::Tensor &gates,
                                    size_t head_dim_index,
                                    size_t total_num_heads,
                                    size_t local_num_heads, size_t tp_rank) {
    if (total_num_heads == local_num_heads) {
        return gates;
    }
    return gates->narrow({{head_dim_index, tp_rank * local_num_heads, local_num_heads}});
}

} // namespace

VideoNSAAttention::VideoNSAAttention(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx, const engine::distributed::RankInfo &rank,
    backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep)
    : infinilm::layers::attention::Attention(runtime, model_config, layer_idx, rank, attention_backend, ep) {
    try {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    total_num_attention_heads_ = total_num_heads;
    tp_rank_ = static_cast<size_t>(rank.tp_rank);
    max_position_embeddings_ = model_config->get<size_t>("max_position_embeddings");

    mrope_section_ = read_mrope_section(model_config);
    mrope_rotary_dim_ = 2 * static_cast<size_t>(mrope_section_[0] + mrope_section_[1] + mrope_section_[2]);
    if (mrope_rotary_dim_ > head_dim_) {
        throw std::runtime_error("VideoNSAAttention: mrope rotary dim exceeds head dim");
    }
    auto mrope_cache = build_mrope_cache(runtime, max_position_embeddings_,
                                         mrope_rotary_dim_,
                                         model_config->get<double>("rope_theta"),
                                         dtype,
                                         runtime.device());
    mrope_cache_runtime_owner_ = runtime.shared_from_this();
    mrope_sin_cache_ = mrope_cache.first;
    mrope_cos_cache_ = mrope_cache.second;
    g_proj_1_ = this->bind_child("g_proj_1", std::make_shared<std::remove_reference_t<decltype(*g_proj_1_ )>>(runtime, hidden_size, hidden_size, true, dtype));
    g_proj_2_ = this->bind_child("g_proj_2", std::make_shared<std::remove_reference_t<decltype(*g_proj_2_ )>>(runtime, hidden_size, 3 * total_num_heads, true, dtype));

    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor VideoNSAAttention::forward(const infinilm::runtime::Tensor &positions,
                                              const infinilm::runtime::Tensor &hidden_states,
                                              global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (mrope_cache_runtime_owner_.get() != &runtime) throw std::invalid_argument("VideoNSA MRoPE cache belongs to another Runtime owner");
    const auto &mm_metadata = forward_context.mm_metadata;
    const bool has_visual_ranges = mm_metadata.visual_token_ranges.has_value() && !mm_metadata.visual_token_ranges->empty();
    const auto &attn_metadata = forward_context.attn_metadata;
    const bool is_paged = ::infinilm::backends::AttentionBackend::PAGED_ATTN == attention_backend_
                       || ::infinilm::backends::AttentionBackend::FLASH_ATTN == attention_backend_;
    const bool is_flash_attn = ::infinilm::backends::AttentionBackend::FLASH_ATTN == attention_backend_;
    const bool has_paged_metadata = attn_metadata.total_sequence_lengths.has_value()
                                 && attn_metadata.slot_mapping.has_value()
                                 && attn_metadata.block_tables.has_value();
    const bool is_decode = has_paged_metadata
                        && hidden_states->size(0) == 1
                        && hidden_states->size(1) == attn_metadata.total_sequence_lengths.value()->shape()[0];
    if (is_paged && (!forward_context.runtime || !forward_context.nsa_owner
        || forward_context.nsa_owner->runtime != forward_context.runtime
        || forward_context.nsa_owner->invocation != nsa_prepared_invocation_))
        throw std::invalid_argument("NSA raw write requires complete worker/model preflight");
    const bool can_use_paged_decode_nsa = is_paged && has_paged_metadata && is_decode;
    if (can_use_paged_decode_nsa && !nsa_prepared_tail_)
        throw std::invalid_argument("NSA one-tail evidence differs from actual model shape");
    const bool can_use_fast_prefill = is_flash_attn
                                   && has_paged_metadata
                                   && hidden_states->size(0) == 1
                                   && !is_decode;
    const bool can_use_paged_prefill_nsa = false && has_visual_ranges
                                        && is_paged
                                        && has_paged_metadata
                                        && hidden_states->size(0) == 1
                                        && hidden_states->size(1) != attn_metadata.total_sequence_lengths.value()->shape()[0];

    const bool can_use_static_scattered_nsa = ::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_;
    if (!can_use_static_scattered_nsa && !can_use_fast_prefill && !can_use_paged_prefill_nsa && !can_use_paged_decode_nsa) {
        return infinilm::layers::attention::Attention::forward(positions, hidden_states, forward_context);
    }

    if (can_use_fast_prefill && !can_use_paged_prefill_nsa) {
        return infinilm::layers::attention::Attention::forward(positions, hidden_states, forward_context);
    }

    auto hidden_states_mutable = hidden_states;
    const size_t batch_size = hidden_states->size(0);
    const size_t seq_len = hidden_states->size(1);
    auto [q, k, v] = qkv_proj_->forward_split(runtime, hidden_states_mutable);

    auto pos_shape = positions->shape();
    infinilm::runtime::Tensor pos_ids_for_rope = positions;
    if (pos_shape.size() != 1 && pos_shape.size() != 2 && pos_shape.size() != 3) {
        throw std::runtime_error("VideoNSAAttention: Unexpected position_ids shape");
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    if (can_use_static_scattered_nsa) {
        auto q_static = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
        auto k_static = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
        auto v_static = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

        auto q_flat = q_static->contiguous(runtime)->view({batch_size * seq_len, num_attention_heads_, head_dim_});
        auto k_flat = k_static->contiguous(runtime)->view({batch_size * seq_len, num_key_value_heads_, head_dim_});
        auto qk_rope = apply_mrope(forward_context.require_runtime(), q_flat,
                                   k_flat,
                                   pos_ids_for_rope,
                                   mrope_cos_cache_.value(),
                                   mrope_sin_cache_.value(),
                                   head_dim_,
                                   mrope_rotary_dim_,
                                   mrope_section_,
                                   mrope_interleaved_);
        auto q_rope = qk_rope.first->view({batch_size, seq_len, num_attention_heads_, head_dim_});
        k_static = qk_rope.second->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

        auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
        auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
        auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
        auto past_lengths = attn_metadata.past_sequence_lengths.value()->contiguous(runtime);
        int32_t past_length = 0;
        runtime.memcpyD2H(&past_length, past_lengths->data(), sizeof(past_length));
        const size_t cache_pos = static_cast<size_t>(past_length);
        const size_t total_seq_len = cache_pos + seq_len;
        k_cache_layer->narrow({{2, cache_pos, seq_len}})->copy_from(runtime, k_static->permute({0, 2, 1, 3}));
        v_cache_layer->narrow({{2, cache_pos, seq_len}})->copy_from(runtime, v_static->permute({0, 2, 1, 3}));
        auto k_total = k_cache_layer->narrow({{2, 0, total_seq_len}});
        auto v_total = v_cache_layer->narrow({{2, 0, total_seq_len}});

        auto gate_hidden = g_proj_1_->forward(runtime, hidden_states_mutable);
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, gate_hidden->shape(), gate_hidden->dtype());
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Silu>(
                runtime, infini::ops::Silu::DefaultConfig(runtime.device().type()), gate_hidden, ops_output);
            gate_hidden = std::move(ops_output);
        }
        auto gates = g_proj_2_->forward(runtime, gate_hidden);
        gates = gates->view({batch_size, seq_len, 3, total_num_attention_heads_});
        gates = local_head_gates(gates, 3, total_num_attention_heads_, num_attention_heads_, tp_rank_);
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, gates->shape(), gates->dtype());
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Sigmoid>(
                runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), gates, ops_output);
            INFINILM_TRACE_OP(Sigmoid, ops_output);
            gates = std::move(ops_output);
        }

        std::vector<infinilm::runtime::Tensor> batch_outputs;
        batch_outputs.reserve(batch_size);
        for (size_t b = 0; b < batch_size; ++b) {
            auto q_b = q_rope->narrow({{0, b, 1}})->squeeze(0);
            auto k_b = k_total->narrow({{0, b, 1}})->squeeze(0)->permute({1, 0, 2});
            auto v_b = v_total->narrow({{0, b, 1}})->squeeze(0)->permute({1, 0, 2});

            auto k_cmp = mean_pool_blocks(runtime, k_b, kNsaBlockSize);
            auto v_cmp = mean_pool_blocks(runtime, v_b, kNsaBlockSize);
            auto comp_heads = grouped_dense_attention(runtime, q_b, k_cmp, v_cmp, num_attention_heads_, num_key_value_heads_, head_dim_, scale);

            const size_t win_len = std::min<size_t>(256, total_seq_len);
            auto k_win = k_b->narrow({{0, total_seq_len - win_len, win_len}});
            auto v_win = v_b->narrow({{0, total_seq_len - win_len, win_len}});
            auto win_heads = grouped_dense_attention(runtime, q_b, k_win, v_win, num_attention_heads_, num_key_value_heads_, head_dim_, scale);

            auto gates_b = gates->narrow({{0, b, 1}});
            auto g_cmp = expand_head_gate(runtime, gates_b->narrow({{2, 0, 1}})->squeeze(2), head_dim_);
            auto g_sel = expand_head_gate(runtime, gates_b->narrow({{2, 1, 1}})->squeeze(2), head_dim_);
            auto g_win = expand_head_gate(runtime, gates_b->narrow({{2, 2, 1}})->squeeze(2), head_dim_);

            // Experimental scattered path: selected-block attention reuses the compressed output
            // to keep this branch expressible with existing dense ops only.
            auto comp_part = infinilm::runtime::Tensor::empty(runtime, comp_heads->shape(), comp_heads->dtype());
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Mul>(
                runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), comp_heads, g_cmp, comp_part);
            INFINILM_TRACE_OP(Mul, comp_part);
            auto sel_part = infinilm::runtime::Tensor::empty(runtime, comp_heads->shape(), comp_heads->dtype());
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Mul>(
                runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), comp_heads, g_sel, sel_part);
            INFINILM_TRACE_OP(Mul, sel_part);
            auto win_part = infinilm::runtime::Tensor::empty(runtime, win_heads->shape(), win_heads->dtype());
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Mul>(
                runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), win_heads, g_win, win_part);
            INFINILM_TRACE_OP(Mul, win_part);
            const auto ops_input = [&]() {
                auto mixed = infinilm::runtime::Tensor::empty(runtime, comp_part->shape(), comp_part->dtype());
                infinilm::execution::submit<infini::ops::Add>(
                    runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), comp_part, sel_part, mixed);
                INFINILM_TRACE_OP(Add, mixed);
                return mixed;
            }();
            auto mixed = infinilm::runtime::Tensor::empty(runtime, ops_input->shape(), ops_input->dtype());
            infinilm::execution::submit<infini::ops::Add>(
                runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), ops_input, win_part, mixed);
            INFINILM_TRACE_OP(Add, mixed);
            batch_outputs.push_back(mixed->view({1, seq_len, num_attention_heads_ * head_dim_}));
        }
        infinilm::runtime::Tensor attn_output;
        if (batch_outputs.size() == 1) {
            attn_output = batch_outputs.front();
        } else {
            attn_output = infinilm::execution::cat_alloc(runtime, batch_outputs, 0);
        }
        return o_proj_->forward(runtime, attn_output);
    }

    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});
    auto qk_rope = apply_mrope(forward_context.require_runtime(), q_reshaped,
                               k_reshaped,
                               pos_ids_for_rope,
                               mrope_cos_cache_.value(),
                               mrope_sin_cache_.value(),
                               head_dim_,
                               mrope_rotary_dim_,
                               mrope_section_,
                               mrope_interleaved_);
    q_reshaped = qk_rope.first;
    k_reshaped = qk_rope.second;

    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    const auto &paged_mapping = attn_metadata.slot_mapping.value();
    const auto paged_config = infinilm::execution::paged_cache_checks::validate(
        runtime, k_cache_layer, v_cache_layer, k_reshaped, v_reshaped, paged_mapping);
    if (paged_config) {
        const auto paged_tokens = paged_mapping->size(0);
        const infini::ops::Tensor paged_scale{nullptr, {1}, infinilm::runtime::DataType::kFloat32, runtime.device()};
        const std::string paged_mode{"auto"};
        for (infinilm::runtime::Size paged_start = 0; paged_start < paged_tokens;) {
            const auto paged_count = std::min<infinilm::runtime::Size>(32768, paged_tokens - paged_start);
            auto paged_key = k_reshaped->narrow({{0, paged_start, paged_count}});
            auto paged_value = v_reshaped->narrow({{0, paged_start, paged_count}});
            auto paged_slots = paged_mapping->narrow({{0, paged_start, paged_count}});
            infinilm::execution::submit<infini::ops::ReshapeAndCacheFlash>(
                runtime, infinilm::execution::RecordingPolicy::AllowHostReplay, *paged_config,
                paged_key, paged_value, paged_slots, paged_scale, paged_scale, paged_mode, k_cache_layer, v_cache_layer);
            INFINILM_TRACE_OP(PagedCaching, k_cache_layer);
            paged_start += paged_count;
        }
    }

    // NSA kernels consume BHSD while the shared paged cache is BSHD.
    auto k_cache_for_nsa = k_cache_layer->permute({0, 2, 1, 3});
    auto v_cache_for_nsa = v_cache_layer->permute({0, 2, 1, 3});

    if (can_use_paged_decode_nsa) {
        auto gate_hidden = g_proj_1_->forward(runtime, hidden_states_mutable);
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, gate_hidden->shape(), gate_hidden->dtype());
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Silu>(
                runtime, infini::ops::Silu::DefaultConfig(runtime.device().type()), gate_hidden, ops_output);
            gate_hidden = std::move(ops_output);
        }
        auto gates = g_proj_2_->forward(runtime, gate_hidden);
        gates = gates->view({seq_len, 3, total_num_attention_heads_});
        gates = local_head_gates(gates, 2, total_num_attention_heads_, num_attention_heads_, tp_rank_);
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, gates->shape(), gates->dtype());
            INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
                && runtime.device().type() != infinilm::runtime::Device::Type::kMars
                && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
            infinilm::execution::submit<infini::ops::Sigmoid>(
                runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), gates, ops_output);
            INFINILM_TRACE_OP(Sigmoid, ops_output);
            gates = std::move(ops_output);
    }

#ifndef INFINICORE_NSA_OPS_UNAVAILABLE
        const auto &nsa_k_cmp = nsa_k_cmp_cache_.value();
        const auto &nsa_v_cmp = nsa_v_cmp_cache_.value();
        const auto &nsa_block_tables = attn_metadata.block_tables.value();
        const auto &nsa_kv_lens = attn_metadata.total_sequence_lengths.value();
        // Temporary pre-write contract until the actual public Ops provider covers it.
        if (!infini::ops::NsaCompressPagedCache::valid_invocation(
                infinilm::execution::detail::view(k_cache_for_nsa),
                infinilm::execution::detail::view(v_cache_for_nsa),
                infinilm::execution::detail::view(nsa_block_tables),
                infinilm::execution::detail::view(nsa_kv_lens),
                static_cast<int>(kNsaBlockSize), nsa_update_last_only_,
                infinilm::execution::detail::view(nsa_k_cmp),
                infinilm::execution::detail::view(nsa_v_cmp)))
            throw std::invalid_argument("NsaCompressPagedCache: invalid storage/alias");
        infinilm::execution::submit<infini::ops::NsaCompressPagedCache>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::NsaCompressPagedCache::DefaultConfig(runtime.device().type()),
            k_cache_for_nsa, v_cache_for_nsa, nsa_block_tables, nsa_kv_lens,
            static_cast<int>(kNsaBlockSize), nsa_update_last_only_, nsa_k_cmp, nsa_v_cmp);
        INFINILM_TRACE_OP(NsaCompressPagedCache, nsa_k_cmp);
#else
        throw std::runtime_error("NsaCompressPagedCache: native InfiniOps provider is unavailable in this build");
#endif
        nsa_lane_.compression_enqueued(); // only successful native submission

        auto nsa_heads = infinilm::runtime::Tensor::empty(runtime, {seq_len, num_attention_heads_, head_dim_}, q_reshaped->dtype());
#ifndef INFINICORE_NSA_OPS_UNAVAILABLE
        // Temporary pre-write contract until the actual public Ops provider covers it.
        if (!infini::ops::NsaPagedAttention::valid_invocation(
                infinilm::execution::detail::view(q_reshaped),
                infinilm::execution::detail::view(nsa_k_cmp),
                infinilm::execution::detail::view(nsa_v_cmp),
                infinilm::execution::detail::view(k_cache_for_nsa),
                infinilm::execution::detail::view(v_cache_for_nsa),
                infinilm::execution::detail::view(nsa_block_tables),
                infinilm::execution::detail::view(nsa_kv_lens),
                infinilm::execution::detail::view(gates),
                scale, static_cast<int>(kNsaBlockSize), int{256}, kNsaSelectBlocks,
                infinilm::execution::detail::view(nsa_heads)))
            throw std::invalid_argument("NsaPagedAttention: invalid storage/alias");
        infinilm::execution::submit<infini::ops::NsaPagedAttention>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::NsaPagedAttention::DefaultConfig(runtime.device().type()),
            q_reshaped, nsa_k_cmp, nsa_v_cmp, k_cache_for_nsa, v_cache_for_nsa,
            nsa_block_tables, nsa_kv_lens, gates, scale,
            static_cast<int>(kNsaBlockSize), int{256}, kNsaSelectBlocks, nsa_heads);
        INFINILM_TRACE_OP(NsaPagedAttention, nsa_heads);
#else
        throw std::runtime_error("NsaPagedAttention: native InfiniOps provider is unavailable in this build");
#endif
        auto attn_output = nsa_heads->view({1, seq_len, num_attention_heads_ * head_dim_});
        return o_proj_->forward(runtime, attn_output);
    }

    auto k_cmp = mean_pool_blocks(runtime, k_reshaped, kNsaBlockSize);
    auto v_cmp = mean_pool_blocks(runtime, v_reshaped, kNsaBlockSize);
    auto nsa_heads = grouped_dense_attention(runtime, q_reshaped, k_cmp, v_cmp, num_attention_heads_, num_key_value_heads_, head_dim_, scale);

    auto gate_hidden = g_proj_1_->forward(runtime, hidden_states_mutable);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, gate_hidden->shape(), gate_hidden->dtype());
        INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
            && runtime.device().type() != infinilm::runtime::Device::Type::kMars
            && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
        infinilm::execution::submit<infini::ops::Silu>(
            runtime, infini::ops::Silu::DefaultConfig(runtime.device().type()), gate_hidden, ops_output);
        gate_hidden = std::move(ops_output);
    }
    auto gates = g_proj_2_->forward(runtime, gate_hidden);
    gates = gates->view({1, seq_len, 3, total_num_attention_heads_});
    gates = local_head_gates(gates, 3, total_num_attention_heads_, num_attention_heads_, tp_rank_);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, gates->shape(), gates->dtype());
        INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
            && runtime.device().type() != infinilm::runtime::Device::Type::kMars
            && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
        infinilm::execution::submit<infini::ops::Sigmoid>(
            runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), gates, ops_output);
        INFINILM_TRACE_OP(Sigmoid, ops_output);
        gates = std::move(ops_output);
    }
    const auto gate_sum_axes = infinilm::execution::tensor_checks::sum::normalized_dim(gates, {2});
    auto gate_sum = infinilm::runtime::Tensor::empty(runtime, infinilm::execution::tensor_checks::sum::reduced_shape(gates, gate_sum_axes, false), gates->dtype());
    infinilm::execution::tensor_checks::sum::validate(runtime, gate_sum, gates, gate_sum_axes, false);
    infinilm::execution::submit<infini::ops::Sum>(runtime, infinilm::execution::RecordingPolicy::DeviceOnly, infini::ops::Sum::DefaultConfig(runtime.device().type()), gates, std::optional<std::vector<int64_t>>{std::vector<int64_t>(gate_sum_axes.begin(), gate_sum_axes.end())}, false, std::optional<infini::ops::DataType>{}, gate_sum); // [1, seq, heads]
    auto gate_expanded = expand_head_gate(runtime, gate_sum, head_dim_);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, nsa_heads->shape(), nsa_heads->dtype());
        INFINILM_ASSERT(runtime.device().type() != infinilm::runtime::Device::Type::kCpu
            && runtime.device().type() != infinilm::runtime::Device::Type::kMars
            && runtime.device().type() != infinilm::runtime::Device::Type::kThead);
        infinilm::execution::submit<infini::ops::Mul>(
            runtime, infini::ops::Mul::DefaultConfig(runtime.device().type()), nsa_heads, gate_expanded, ops_output);
        INFINILM_TRACE_OP(Mul, ops_output);
        nsa_heads = std::move(ops_output);
    }

    auto attn_output = nsa_heads->view({1, seq_len, num_attention_heads_ * head_dim_});
    return o_proj_->forward(runtime, attn_output);
}

void VideoNSAAttention::prepare_paged_owner(global_state::ForwardContext &context) const {
    if (!context.runtime || !context.nsa_owner || context.nsa_owner->runtime != context.runtime)
        throw std::invalid_argument("VideoNSA requires worker-scoped owner proof");
    auto &runtime = *context.runtime;
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording() || runtime.device().type() != infinilm::runtime::Device::Type::kNvidia
        || (attention_backend_ != backends::AttentionBackend::PAGED_ATTN
            && attention_backend_ != backends::AttentionBackend::FLASH_ATTN))
        throw std::invalid_argument("VideoNSA owner slice requires NVIDIA eager paged attention");
    const auto &proof = *context.nsa_owner;
    const auto &s = proof.owner;
    if (!num_key_value_heads_ || !num_attention_heads_ || num_key_value_heads_ > 65535
        || num_attention_heads_ > 65535 || num_attention_heads_ % num_key_value_heads_)
        throw std::invalid_argument("VideoNSA local heads outside native launch domain");
    if (layer_idx_ >= context.kv_cache_vec.size() || !context.kv_cache_vec[layer_idx_])
        throw std::invalid_argument("VideoNSA raw backing absent");
    const auto &raw = context.kv_cache_vec[layer_idx_];
    if (raw->device() != runtime.device() || raw->ndim() != 5 || !raw->is_contiguous() || raw->size(0) != 2
        || raw->size(1) != s.page_count || raw->size(2) != s.page_size
        || raw->size(3) != num_key_value_heads_ || raw->size(4) != head_dim_ || head_dim_ != 128
        || (raw->dtype() != infinilm::runtime::DataType::kFloat16 && raw->dtype() != infinilm::runtime::DataType::kBFloat16))
        throw std::invalid_argument("VideoNSA raw backing/view does not match owner geometry");
    if (nsa_backing_epoch_ == s.pool_epoch && nsa_raw_lease_) {
        const auto &old = nsa_raw_lease_.value();
        if (old->data() != raw->data() || old->shape() != raw->shape() || old->strides() != raw->strides()
            || old->dtype() != raw->dtype() || old->device() != raw->device())
            throw std::invalid_argument("VideoNSA raw view changed without pool rebind");
    } else {
        // A new worker-bound epoch is published only AFTER actual old-owner drain.
        // Strong old references survive revoke and are released only at this point.
        nsa_lane_.revoke();
        nsa_k_cmp_cache_.reset(); nsa_v_cmp_cache_.reset(); nsa_raw_lease_.reset();
        nsa_backing_epoch_ = s.pool_epoch;
    }
    nsa_raw_lease_ = raw;
    nsa_prepared_invocation_ = proof.invocation;
    nsa_prepared_tail_ = false;
    if (!proof.one_tail) { nsa_lane_.revoke(); return; } // prefill writes invalidate, never seed
    const auto cmp_blocks = s.page_count * (s.page_size / 64);
    const bool need = !nsa_k_cmp_cache_ || !nsa_v_cmp_cache_
        || nsa_k_cmp_cache_.value()->shape() != infinilm::runtime::Shape{cmp_blocks, num_key_value_heads_, head_dim_}
        || nsa_k_cmp_cache_.value()->dtype() != raw->dtype()
        || nsa_v_cmp_cache_.value()->shape() != infinilm::runtime::Shape{cmp_blocks, num_key_value_heads_, head_dim_}
        || nsa_v_cmp_cache_.value()->dtype() != raw->dtype()
        || nsa_k_cmp_cache_.value()->device() != runtime.device()
        || nsa_v_cmp_cache_.value()->device() != runtime.device();
    if (need) {
        if (nsa_lane_.status() == nsa_serial::Status::QueuedOnOwnerStream)
            throw std::logic_error("VideoNSA compressed replacement requires actual owner drain");
        if (nsa_compressed_generation_ == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("VideoNSA compressed generation exhausted");
        ++nsa_compressed_generation_;
        nsa_k_cmp_cache_ = infinilm::runtime::Tensor::empty(runtime, {cmp_blocks, num_key_value_heads_, head_dim_}, raw->dtype());
        nsa_v_cmp_cache_ = infinilm::runtime::Tensor::empty(runtime, {cmp_blocks, num_key_value_heads_, head_dim_}, raw->dtype());
    }
    nsa_serial::Evidence evidence;
    evidence.binding = {s.pool_epoch, proof.worker, proof.worker, s.pool_epoch,
                        nsa_compressed_generation_, s.page_count, s.page_size, 64};
    for (const auto &r : s.rows) {
        nsa_serial::Row row{r.incarnation, r.revision, r.past, r.total,
                           static_cast<std::uint64_t>(r.slots.front()), {}};
        for (const auto &p : r.pages) row.pages.push_back({p.first, p.second});
        evidence.rows.push_back(std::move(row));
    }
    nsa_update_last_only_ = nsa_lane_.begin(std::move(evidence)); // BEFORE ANY layer raw write
    if (!nsa_update_last_only_ && s.upload.table_width > 65535 / (s.page_size / 64))
        throw std::invalid_argument("VideoNSA FULL uploaded width outside native launch domain");
    nsa_prepared_tail_ = true;
}

void VideoNSAAttention::finish_paged_owner(bool completed) const {
    if (nsa_prepared_tail_)
        nsa_lane_.finish(completed ? nsa_serial::Boundary::ExistingOwnerStreamSync
                                   : nsa_serial::Boundary::HostSubmissionOnly);
    nsa_prepared_tail_ = false; nsa_prepared_invocation_ = 0;
}

void VideoNSAAttention::invalidate_paged_owner() const noexcept {
    try { nsa_lane_.revoke(); }
    catch (...) {
        // A rejected foreign mutation never releases/promotes storage. Worker
        // failure remains terminal and performs the actual Runtime drain.
    }
    nsa_prepared_tail_ = false; nsa_prepared_invocation_ = 0;
    // Keep raw/compressed Tensor leases even when CPU validity is absent.
}

} // namespace infinilm::models::videonsa
