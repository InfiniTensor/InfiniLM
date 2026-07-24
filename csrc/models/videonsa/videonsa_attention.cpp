#include "videonsa_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/mrope.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/nsa_compress_paged_cache.hpp"
#include "infinicore/ops/nsa_paged_attention.hpp"
#include "infinicore/ops/paged_caching.hpp"
#include "infinicore/ops/sigmoid.hpp"
#include "infinicore/ops/silu.hpp"
#include "infinicore/ops/softmax.hpp"
#include "infinicore/ops/sum.hpp"
#include "infinicore/ops/take.hpp"
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

infinicore::Tensor scalar_tensor(float value, const infinicore::Device &device) {
    auto cpu = infinicore::Tensor::from_blob(&value, {1}, infinicore::DataType::kFloat32, infinicore::Device{infinicore::Device::Type::kCpu});
    return cpu->to(device);
}

// Temporarily creating cache here. Scheduled to be replaced by a revamped mrope module.
std::pair<infinicore::Tensor, infinicore::Tensor> build_mrope_cache(size_t max_seq_len,
                                                                    size_t rotary_dim,
                                                                    double theta,
                                                                    const infinicore::DataType &dtype,
                                                                    const infinicore::Device &device) {
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

    const auto cpu = infinicore::Device{infinicore::Device::Type::kCpu};
    auto sin_cache = infinicore::Tensor::empty({max_seq_len, cache_dim}, dtype, device);
    auto cos_cache = infinicore::Tensor::empty({max_seq_len, cache_dim}, dtype, device);
    if (dtype == infinicore::DataType::kFloat32) {
        auto sin_cpu = infinicore::Tensor::from_blob(sin_data.data(), {max_seq_len, cache_dim}, infinicore::DataType::kFloat32, cpu);
        auto cos_cpu = infinicore::Tensor::from_blob(cos_data.data(), {max_seq_len, cache_dim}, infinicore::DataType::kFloat32, cpu);
        sin_cache->copy_from(sin_cpu);
        cos_cache->copy_from(cos_cpu);
        return {sin_cache, cos_cache};
    }
    if (dtype == infinicore::DataType::kBFloat16) {
        std::vector<uint16_t> sin_bf16(numel);
        std::vector<uint16_t> cos_bf16(numel);
        for (size_t i = 0; i < numel; ++i) {
            sin_bf16[i] = f32_to_bf16(sin_data[i]);
            cos_bf16[i] = f32_to_bf16(cos_data[i]);
        }
        auto sin_cpu = infinicore::Tensor::from_blob(sin_bf16.data(), {max_seq_len, cache_dim}, infinicore::DataType::kBFloat16, cpu);
        auto cos_cpu = infinicore::Tensor::from_blob(cos_bf16.data(), {max_seq_len, cache_dim}, infinicore::DataType::kBFloat16, cpu);
        sin_cache->copy_from(sin_cpu);
        cos_cache->copy_from(cos_cpu);
        return {sin_cache, cos_cache};
    }
    if (dtype == infinicore::DataType::kFloat16) {
        std::vector<uint16_t> sin_f16(numel);
        std::vector<uint16_t> cos_f16(numel);
        for (size_t i = 0; i < numel; ++i) {
            sin_f16[i] = f32_to_f16(sin_data[i]);
            cos_f16[i] = f32_to_f16(cos_data[i]);
        }
        auto sin_cpu = infinicore::Tensor::from_blob(sin_f16.data(), {max_seq_len, cache_dim}, infinicore::DataType::kFloat16, cpu);
        auto cos_cpu = infinicore::Tensor::from_blob(cos_f16.data(), {max_seq_len, cache_dim}, infinicore::DataType::kFloat16, cpu);
        sin_cache->copy_from(sin_cpu);
        cos_cache->copy_from(cos_cpu);
        return {sin_cache, cos_cache};
    }
    throw std::runtime_error("VideoNSAAttention: mrope cache dtype is unsupported");
}

std::pair<infinicore::Tensor, infinicore::Tensor> apply_mrope(const infinicore::Tensor &q,
                                                              const infinicore::Tensor &k,
                                                              const infinicore::Tensor &positions,
                                                              const infinicore::Tensor &cos_cache,
                                                              const infinicore::Tensor &sin_cache,
                                                              size_t head_dim,
                                                              size_t rotary_dim,
                                                              const std::array<int, 3> &section,
                                                              bool interleaved) {
    const size_t num_tokens = q->size(0);
    auto q_flat = q->contiguous()->view({num_tokens, q->size(1) * head_dim});
    auto k_flat = k->contiguous()->view({num_tokens, k->size(1) * head_dim});
    auto qk_rope = infinicore::op::mrope(q_flat,
                                         k_flat,
                                         cos_cache,
                                         sin_cache,
                                         positions,
                                         static_cast<int>(head_dim),
                                         static_cast<int>(rotary_dim),
                                         section[0],
                                         section[1],
                                         section[2],
                                         interleaved);
    return {qk_rope.first->view(q->shape()), qk_rope.second->view(k->shape())};
}

infinicore::Tensor mean_pool_blocks(const infinicore::Tensor &x, size_t block_size) {
    const size_t seq_len = x->size(0);
    std::vector<infinicore::Tensor> blocks;
    blocks.reserve((seq_len + block_size - 1) / block_size);
    for (size_t start = 0; start < seq_len; start += block_size) {
        const size_t len = std::min(block_size, seq_len - start);
        auto block = x->narrow({{0, start, len}});
        auto pooled = infinicore::op::sum(block, {0}, false);
        blocks.push_back(pooled->unsqueeze(0));
    }
    return blocks.size() == 1 ? blocks.front() : infinicore::op::cat(blocks, 0);
}

infinicore::Tensor repeat_group_tensor(const infinicore::Tensor &x, size_t repeats) {
    std::vector<infinicore::Tensor> copies;
    copies.reserve(repeats);
    for (size_t i = 0; i < repeats; ++i) {
        copies.push_back(x);
    }
    return copies.size() == 1 ? copies.front() : infinicore::op::cat(copies, 0);
}

infinicore::Tensor grouped_dense_attention(const infinicore::Tensor &q,
                                           const infinicore::Tensor &k,
                                           const infinicore::Tensor &v,
                                           size_t num_heads,
                                           size_t num_kv_heads,
                                           size_t head_dim,
                                           float scale) {
    const size_t seq_len = q->size(0);
    const size_t kv_len = k->size(0);
    const size_t heads_per_group = num_heads / num_kv_heads;
    std::vector<infinicore::Tensor> group_outputs;
    group_outputs.reserve(num_kv_heads);
    for (size_t g = 0; g < num_kv_heads; ++g) {
        auto q_group = q->narrow({{1, g * heads_per_group, heads_per_group}})
                           ->permute({1, 0, 2})
                           ->contiguous();                               // [heads_per_group, seq, dim]
        auto k_group = k->narrow({{1, g, 1}})->squeeze(1)->unsqueeze(0); // [1, kv, dim]
        auto v_group = v->narrow({{1, g, 1}})->squeeze(1)->unsqueeze(0); // [1, kv, dim]
        auto k_repeated = repeat_group_tensor(k_group, heads_per_group);
        auto v_repeated = repeat_group_tensor(v_group, heads_per_group);
        auto scores = infinicore::op::matmul(q_group, k_repeated->permute({0, 2, 1}), scale);
        infinicore::op::softmax_(scores, scores, -1);
        auto out = infinicore::op::matmul(scores, v_repeated)
                       ->view({heads_per_group, seq_len, head_dim})
                       ->permute({1, 0, 2})
                       ->contiguous();
        group_outputs.push_back(out);
    }
    return group_outputs.size() == 1 ? group_outputs.front() : infinicore::op::cat(group_outputs, 1);
}

infinicore::Tensor expand_head_gate(const infinicore::Tensor &gate, size_t head_dim) {
    const size_t seq_len = gate->size(1);
    const size_t num_heads = gate->size(2);
    auto flat_gate = gate->contiguous()->view({seq_len * num_heads});
    std::vector<int64_t> expand_indices(seq_len * num_heads * head_dim);
    for (size_t row = 0; row < seq_len * num_heads; ++row) {
        for (size_t d = 0; d < head_dim; ++d) {
            expand_indices[row * head_dim + d] = static_cast<int64_t>(row);
        }
    }
    auto indices = infinicore::Tensor::empty({seq_len, num_heads, head_dim}, infinicore::DataType::kInt64, gate->device());
    infinicore::context::memcpyH2D(indices->data(), expand_indices.data(), expand_indices.size() * sizeof(int64_t), false);
    return infinicore::op::take(flat_gate, indices);
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

infinicore::Tensor local_head_gates(const infinicore::Tensor &gates,
                                    size_t head_dim_index,
                                    size_t total_num_heads,
                                    size_t local_num_heads) {
    if (total_num_heads == local_num_heads) {
        return gates;
    }
    const size_t tp_rank = static_cast<size_t>(infinilm::global_state::get_tensor_model_parallel_rank());
    return gates->narrow({{head_dim_index, tp_rank * local_num_heads, local_num_heads}});
}

} // namespace

VideoNSAAttention::VideoNSAAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device)
    : infinilm::layers::attention::Attention(model_config, layer_idx, device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    total_num_attention_heads_ = total_num_heads;
    max_position_embeddings_ = model_config->get<size_t>("max_position_embeddings");

    mrope_section_ = read_mrope_section(model_config);
    mrope_rotary_dim_ = 2 * static_cast<size_t>(mrope_section_[0] + mrope_section_[1] + mrope_section_[2]);
    if (mrope_rotary_dim_ > head_dim_) {
        throw std::runtime_error("VideoNSAAttention: mrope rotary dim exceeds head dim");
    }
    auto mrope_cache = build_mrope_cache(max_position_embeddings_,
                                         mrope_rotary_dim_,
                                         model_config->get<double>("rope_theta"),
                                         dtype,
                                         device);
    mrope_sin_cache_ = mrope_cache.first;
    mrope_cos_cache_ = mrope_cache.second;
    INFINICORE_NN_MODULE_INIT(g_proj_1, hidden_size, hidden_size, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(g_proj_2, hidden_size, 3 * total_num_heads, true, dtype, device);
}

infinicore::Tensor VideoNSAAttention::forward(const infinicore::Tensor &positions,
                                              const infinicore::Tensor &hidden_states) const {
    const auto &forward_context = infinilm::global_state::get_forward_context();
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
    const bool can_use_paged_decode_nsa = is_paged && has_paged_metadata && is_decode;
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
        return infinilm::layers::attention::Attention::forward(positions, hidden_states);
    }

    if (can_use_fast_prefill && !can_use_paged_prefill_nsa) {
        return infinilm::layers::attention::Attention::forward(positions, hidden_states);
    }

    auto hidden_states_mutable = hidden_states;
    const size_t batch_size = hidden_states->size(0);
    const size_t seq_len = hidden_states->size(1);
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    auto pos_shape = positions->shape();
    infinicore::Tensor pos_ids_for_rope = positions;
    if (pos_shape.size() != 1 && pos_shape.size() != 2 && pos_shape.size() != 3) {
        throw std::runtime_error("VideoNSAAttention: Unexpected position_ids shape");
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    if (can_use_static_scattered_nsa) {
        auto q_static = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
        auto k_static = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
        auto v_static = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

        auto q_flat = q_static->contiguous()->view({batch_size * seq_len, num_attention_heads_, head_dim_});
        auto k_flat = k_static->contiguous()->view({batch_size * seq_len, num_key_value_heads_, head_dim_});
        auto qk_rope = apply_mrope(q_flat,
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
        const size_t cache_pos = reinterpret_cast<int32_t *>(attn_metadata.past_sequence_lengths.value()->to(infinicore::Device{infinicore::Device::Type::kCpu})->data())[0];
        const size_t total_seq_len = cache_pos + seq_len;
        k_cache_layer->narrow({{2, cache_pos, seq_len}})->copy_from(k_static->permute({0, 2, 1, 3}));
        v_cache_layer->narrow({{2, cache_pos, seq_len}})->copy_from(v_static->permute({0, 2, 1, 3}));
        auto k_total = k_cache_layer->narrow({{2, 0, total_seq_len}});
        auto v_total = v_cache_layer->narrow({{2, 0, total_seq_len}});

        auto gate_hidden = g_proj_1_->forward(hidden_states_mutable);
        gate_hidden = infinicore::op::silu(gate_hidden);
        auto gates = g_proj_2_->forward(gate_hidden);
        gates = gates->view({batch_size, seq_len, 3, total_num_attention_heads_});
        gates = local_head_gates(gates, 3, total_num_attention_heads_, num_attention_heads_);
        gates = infinicore::op::sigmoid(gates);

        std::vector<infinicore::Tensor> batch_outputs;
        batch_outputs.reserve(batch_size);
        for (size_t b = 0; b < batch_size; ++b) {
            auto q_b = q_rope->narrow({{0, b, 1}})->squeeze(0);
            auto k_b = k_total->narrow({{0, b, 1}})->squeeze(0)->permute({1, 0, 2});
            auto v_b = v_total->narrow({{0, b, 1}})->squeeze(0)->permute({1, 0, 2});

            auto k_cmp = mean_pool_blocks(k_b, kNsaBlockSize);
            auto v_cmp = mean_pool_blocks(v_b, kNsaBlockSize);
            auto comp_heads = grouped_dense_attention(q_b, k_cmp, v_cmp, num_attention_heads_, num_key_value_heads_, head_dim_, scale);

            const size_t win_len = std::min<size_t>(256, total_seq_len);
            auto k_win = k_b->narrow({{0, total_seq_len - win_len, win_len}});
            auto v_win = v_b->narrow({{0, total_seq_len - win_len, win_len}});
            auto win_heads = grouped_dense_attention(q_b, k_win, v_win, num_attention_heads_, num_key_value_heads_, head_dim_, scale);

            auto gates_b = gates->narrow({{0, b, 1}});
            auto g_cmp = expand_head_gate(gates_b->narrow({{2, 0, 1}})->squeeze(2), head_dim_);
            auto g_sel = expand_head_gate(gates_b->narrow({{2, 1, 1}})->squeeze(2), head_dim_);
            auto g_win = expand_head_gate(gates_b->narrow({{2, 2, 1}})->squeeze(2), head_dim_);

            // Experimental scattered path: selected-block attention reuses the compressed output
            // to keep this branch expressible with existing dense ops only.
            auto comp_part = infinicore::op::mul(comp_heads, g_cmp);
            auto sel_part = infinicore::op::mul(comp_heads, g_sel);
            auto win_part = infinicore::op::mul(win_heads, g_win);
            auto mixed = infinicore::op::add(infinicore::op::add(comp_part, sel_part), win_part);
            batch_outputs.push_back(mixed->view({1, seq_len, num_attention_heads_ * head_dim_}));
        }
        auto attn_output = batch_outputs.size() == 1 ? batch_outputs.front() : infinicore::op::cat(batch_outputs, 0);
        return o_proj_->forward(attn_output);
    }

    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});
    auto qk_rope = apply_mrope(q_reshaped,
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
    auto k_cache_for_nsa = is_flash_attn ? k_cache_layer->permute({0, 2, 1, 3}) : k_cache_layer;
    auto v_cache_for_nsa = is_flash_attn ? v_cache_layer->permute({0, 2, 1, 3}) : v_cache_layer;

    infinicore::op::paged_caching_(k_cache_for_nsa, v_cache_for_nsa, k_reshaped, v_reshaped, attn_metadata.slot_mapping.value());

    if (can_use_paged_decode_nsa) {
        auto gate_hidden = g_proj_1_->forward(hidden_states_mutable);
        gate_hidden = infinicore::op::silu(gate_hidden);
        auto gates = g_proj_2_->forward(gate_hidden);
        gates = gates->view({seq_len, 3, total_num_attention_heads_});
        gates = local_head_gates(gates, 2, total_num_attention_heads_, num_attention_heads_);
        gates = infinicore::op::sigmoid(gates);

        const size_t page_block_size = k_cache_for_nsa->size(2);
        const size_t subblocks_per_page = page_block_size / kNsaBlockSize;
        const size_t cmp_blocks = k_cache_for_nsa->size(0) * subblocks_per_page;
        const bool need_cmp_alloc = !nsa_k_cmp_cache_.has_value()
                                 || nsa_k_cmp_cache_.value()->size(0) != cmp_blocks
                                 || nsa_k_cmp_cache_.value()->size(1) != num_key_value_heads_
                                 || nsa_k_cmp_cache_.value()->size(2) != head_dim_;
        if (need_cmp_alloc) {
            nsa_k_cmp_cache_ = infinicore::Tensor::empty({cmp_blocks, num_key_value_heads_, head_dim_}, k_cache_for_nsa->dtype(), k_cache_layer->device());
            nsa_v_cmp_cache_ = infinicore::Tensor::empty({cmp_blocks, num_key_value_heads_, head_dim_}, v_cache_for_nsa->dtype(), v_cache_layer->device());
            nsa_cmp_cache_ready_ = false;
        }
        const bool update_last_only = nsa_cmp_cache_ready_;
        infinicore::op::nsa_compress_paged_cache_(
            nsa_k_cmp_cache_.value(),
            nsa_v_cmp_cache_.value(),
            k_cache_for_nsa,
            v_cache_for_nsa,
            attn_metadata.block_tables.value(),
            attn_metadata.total_sequence_lengths.value(),
            static_cast<int>(kNsaBlockSize),
            update_last_only);
        nsa_cmp_cache_ready_ = true;

        auto nsa_heads = infinicore::Tensor::empty({seq_len, num_attention_heads_, head_dim_}, q_reshaped->dtype(), q_reshaped->device());
        infinicore::op::nsa_paged_attention_(
            nsa_heads,
            q_reshaped,
            nsa_k_cmp_cache_.value(),
            nsa_v_cmp_cache_.value(),
            k_cache_for_nsa,
            v_cache_for_nsa,
            attn_metadata.block_tables.value(),
            attn_metadata.total_sequence_lengths.value(),
            gates,
            scale,
            static_cast<int>(kNsaBlockSize),
            256,
            kNsaSelectBlocks);
        auto attn_output = nsa_heads->view({1, seq_len, num_attention_heads_ * head_dim_});
        return o_proj_->forward(attn_output);
    }

    auto k_cmp = mean_pool_blocks(k_reshaped, kNsaBlockSize);
    auto v_cmp = mean_pool_blocks(v_reshaped, kNsaBlockSize);
    auto nsa_heads = grouped_dense_attention(q_reshaped, k_cmp, v_cmp, num_attention_heads_, num_key_value_heads_, head_dim_, scale);

    auto gate_hidden = g_proj_1_->forward(hidden_states_mutable);
    gate_hidden = infinicore::op::silu(gate_hidden);
    auto gates = g_proj_2_->forward(gate_hidden);
    gates = gates->view({1, seq_len, 3, total_num_attention_heads_});
    gates = local_head_gates(gates, 3, total_num_attention_heads_, num_attention_heads_);
    gates = infinicore::op::sigmoid(gates);
    auto gate_sum = infinicore::op::sum(gates, {2}, false); // [1, seq, heads]
    auto gate_expanded = expand_head_gate(gate_sum, head_dim_);
    nsa_heads = infinicore::op::mul(nsa_heads, gate_expanded);

    auto attn_output = nsa_heads->view({1, seq_len, num_attention_heads_ * head_dim_});
    return o_proj_->forward(attn_output);
}

void VideoNSAAttention::process_weights_after_loading() {
    infinilm::layers::attention::Attention::process_weights_after_loading();
    g_proj_1_->process_weights_after_loading();
    g_proj_2_->process_weights_after_loading();
}

void VideoNSAAttention::reset_runtime_state() const {
    infinilm::layers::attention::Attention::reset_runtime_state();
    g_proj_1_->reset_runtime_state();
    g_proj_2_->reset_runtime_state();
    nsa_k_cmp_cache_.reset();
    nsa_v_cmp_cache_.reset();
    nsa_cmp_cache_ready_ = false;
}

} // namespace infinilm::models::videonsa
