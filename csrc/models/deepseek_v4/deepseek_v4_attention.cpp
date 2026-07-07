#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/unweighted_rms_norm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

void warn_attention_approximation_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        spdlog::warn(
            "DeepseekV4Attention: sliding layers use a hybrid GPU/CPU path; "
            "compressed layers (CSA/HCA) still use the CPU reference attention.");
    });
}

void write_paged_kv_cache_(size_t layer_idx,
                           const infinicore::Tensor &key_states,
                           size_t seq_len,
                           size_t num_kv_heads,
                           size_t head_dim) {
    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &attn_metadata = forward_context.attn_metadata;
    if (!attn_metadata.slot_mapping.has_value()) {
        return;
    }
    if (layer_idx >= forward_context.kv_cache_vec.size()) {
        throw std::runtime_error("DeepseekV4Attention: kv_cache_vec is not initialized for paged attention");
    }

    auto &kv_cache = forward_context.kv_cache_vec[layer_idx];
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    auto kv_paged = key_states->view({seq_len, num_kv_heads, head_dim});
    infinicore::op::paged_caching_(
        k_cache_layer,
        v_cache_layer,
        kv_paged,
        kv_paged,
        attn_metadata.slot_mapping.value());
}

void apply_compress_block_rope(std::vector<float> &kv_comp,
                               size_t batch_size,
                               size_t nb,
                               size_t head_dim,
                               size_t seq_len,
                               const std::vector<int64_t> &positions,
                               size_t compress_ratio,
                               const DeepseekV4RopeParams &params) {
    if (compress_ratio == 0 || nb == 0) {
        return;
    }
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t block = 0; block < nb; ++block) {
            const size_t block_token = std::min(block * compress_ratio, seq_len > 0 ? seq_len - 1 : 0);
            const int64_t block_pos = (positions[block_token] / static_cast<int64_t>(compress_ratio))
                                    * static_cast<int64_t>(compress_ratio);
            const size_t kv_offset = (b * nb + block) * head_dim;
            apply_rope_at_offset(kv_comp, kv_offset, block_pos, params, false);
        }
    }
}

} // namespace

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------

DeepseekV4Attention::DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         const infinicore::Device &device)
    : DeepseekV4Attention(std::move(model_config), 0, device) {
}

DeepseekV4Attention::DeepseekV4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                         size_t layer_idx,
                                         const infinicore::Device &device)
    : layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      global_num_attention_heads_(model_config->get<size_t>("num_attention_heads")),
      num_attention_heads_(global_num_attention_heads_),
      num_key_value_heads_(model_config->get_or<size_t>("num_key_value_heads", 1)),
      head_dim_(model_config->get<size_t>("head_dim")),
      q_lora_rank_(model_config->get<size_t>("q_lora_rank")),
      o_lora_rank_(model_config->get<size_t>("o_lora_rank")),
      global_o_groups_(model_config->get<size_t>("o_groups")),
      o_groups_(global_o_groups_),
      o_a_input_size_(global_num_attention_heads_ * head_dim_ / global_o_groups_),
      o_a_output_size_(o_lora_rank_ * global_o_groups_),
      sliding_window_(model_config->get_or<size_t>("sliding_window", 0)),
      rms_norm_eps_(model_config->get<double>("rms_norm_eps")),
      rotary_emb_(model_config, layer_idx, device),
      softmax_scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    const auto &dtype = model_config->get_dtype();
    const size_t q_output_size = num_attention_heads_ * head_dim_;
    const size_t compress_ratio = rotary_emb_.compress_ratio();
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto quantization_method = deepseek_v4_linear_quantization(model_config, true);
    auto none_quantization = deepseek_v4_linear_quantization(model_config, false);

    INFINICORE_NN_MODULE_INIT(q_norm, q_lora_rank_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(wq_a, hidden_size_, q_lora_rank_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wq_b, q_lora_rank_, q_output_size, quantization_method, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);

    INFINICORE_NN_MODULE_INIT(kv_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(wkv, hidden_size_, head_dim_, quantization_method, false, dtype, device);

    INFINICORE_NN_MODULE_INIT(wo_a, o_a_input_size_, o_a_output_size_, none_quantization, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(wo_b, o_a_output_size_, hidden_size_, quantization_method, false, dtype, device, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);

    if (compress_ratio == 4) {
        INFINICORE_NN_MODULE_INIT(indexer, model_config, compress_ratio, device);
    }
    if (compress_ratio > 1) {
        INFINICORE_NN_MODULE_INIT(compressor, model_config, compress_ratio, head_dim_, device);
    }

    const int tp_size = rank_info.tp_size;
    if (num_attention_heads_ % static_cast<size_t>(tp_size) != 0) {
        throw std::runtime_error("DeepseekV4Attention: num_attention_heads must be divisible by tp_size");
    }
    if (global_o_groups_ % static_cast<size_t>(tp_size) != 0) {
        throw std::runtime_error("DeepseekV4Attention: o_groups must be divisible by tp_size");
    }
    num_attention_heads_ /= static_cast<size_t>(tp_size);
    o_groups_ = global_o_groups_ / static_cast<size_t>(tp_size);
    o_a_output_size_ = o_lora_rank_ * o_groups_;
    if (num_key_value_heads_ >= static_cast<size_t>(tp_size)) {
        num_key_value_heads_ /= static_cast<size_t>(tp_size);
    } else {
        num_key_value_heads_ = 1;
    }
    if (num_attention_heads_ % o_groups_ != 0) {
        throw std::runtime_error("DeepseekV4Attention: local num_attention_heads must be divisible by local o_groups");
    }

    INFINICORE_NN_PARAMETER_INIT(attn_sink, ({global_num_attention_heads_}, infinicore::DataType::F32, device,
                                             0, rank_info.tp_rank, rank_info.tp_size));

    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, softmax_scale_, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);
    infinilm::layers::attention::init_kv_cache_quant_params(
        register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

// -----------------------------------------------------------------------------
// Forward entry
// -----------------------------------------------------------------------------

infinicore::Tensor DeepseekV4Attention::forward(const infinicore::Tensor &positions,
                                                const infinicore::Tensor &hidden_states) const {
    warn_attention_approximation_once();
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor DeepseekV4Attention::forward_static_(const infinicore::Tensor &positions,
                                                        const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    const auto qk = project_qk_rope_(positions, hidden_states, batch_size, seq_len);
    const AttentionInputs inputs{
        position_ids_as_vector(qk.pos_ids),
        qk.q_normed,
        qk.key_states,
        hidden_states,
        qk.q_residual,
        0,
    };
    return apply_grouped_output_projection_(run_attention_(inputs));
}

infinicore::Tensor DeepseekV4Attention::forward_paged_(const infinicore::Tensor &positions,
                                                       const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);

    const auto qk = project_qk_rope_(positions, hidden_states, batch_size, seq_len);
    const bool is_decode = is_paged_decode_step_(seq_len);
    const auto inputs = build_paged_attention_inputs_(qk, hidden_states, seq_len, is_decode);
    return apply_grouped_output_projection_(run_attention_(inputs));
}

// -----------------------------------------------------------------------------
// Q/K projection + RoPE
// -----------------------------------------------------------------------------

DeepseekV4Attention::QkProjections DeepseekV4Attention::project_qk_rope_(
    const infinicore::Tensor &positions,
    const infinicore::Tensor &hidden_states,
    size_t batch_size,
    size_t seq_len) const {
    const auto pos_ids = position_ids_for_rope(positions, seq_len);
    auto hidden_states_mut = hidden_states;

    auto q_residual = q_norm_->forward(wq_a_->forward(hidden_states_mut));
    auto q = wq_b_->forward(q_residual)->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto q_normed = infinicore::op::unweighted_rms_norm(q->contiguous(), static_cast<float>(rms_norm_eps_));

    auto kv = kv_norm_->forward(wkv_->forward(hidden_states_mut))
                  ->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    QkProjections result;
    result.q_residual = std::move(q_residual);
    result.pos_ids = pos_ids;
    std::tie(result.q_normed, result.key_states) = rotary_emb_.forward(q_normed, kv, pos_ids);
    return result;
}

// -----------------------------------------------------------------------------
// Paged decode cache management
// -----------------------------------------------------------------------------

bool DeepseekV4Attention::is_paged_decode_step_(size_t seq_len) const {
    if (seq_len != 1 || cached_seq_len_ == 0) {
        return false;
    }
    const auto &attn_metadata = infinilm::global_state::get_forward_context().attn_metadata;
    if (!attn_metadata.past_sequence_lengths.has_value()) {
        return false;
    }
    const auto past_lengths = tensor_to_int64_vector(attn_metadata.past_sequence_lengths.value());
    return !past_lengths.empty() && past_lengths[0] > 0;
}

DeepseekV4Attention::AttentionInputs DeepseekV4Attention::build_paged_attention_inputs_(
    const QkProjections &qk,
    const infinicore::Tensor &hidden_states,
    size_t seq_len,
    bool is_decode) const {

    write_paged_kv_cache_(layer_idx_, qk.key_states, seq_len, num_key_value_heads_, head_dim_);

    if (is_decode) {
        cached_key_states_ = infinicore::op::cat({cached_key_states_, qk.key_states}, 1);
    } else {
        cached_key_states_ = qk.key_states;
    }

    const auto pos = position_ids_as_vector(qk.pos_ids);
    AttentionInputs inputs;
    inputs.q_normed = qk.q_normed;
    inputs.key_states = cached_key_states_;
    inputs.q_residual = qk.q_residual;

    if (is_decode) {
        inputs.query_start = cached_seq_len_;
        cached_hidden_states_ = infinicore::op::cat({cached_hidden_states_, hidden_states}, 1);
        cached_q_residual_ = infinicore::op::cat({cached_q_residual_, qk.q_residual}, 1);
        cached_positions_.insert(cached_positions_.end(), pos.begin(), pos.end());
        cached_seq_len_ = cached_positions_.size();
        inputs.positions = cached_positions_;
        inputs.hidden_states = cached_hidden_states_;
    } else {
        cached_hidden_states_ = hidden_states;
        cached_q_residual_ = qk.q_residual;
        cached_positions_ = pos;
        cached_seq_len_ = seq_len;
        inputs.positions = pos;
        inputs.hidden_states = hidden_states;
    }
    return inputs;
}

// -----------------------------------------------------------------------------
// Attention dispatch
// -----------------------------------------------------------------------------

infinicore::Tensor DeepseekV4Attention::run_attention_(const AttentionInputs &inputs) const {
    return dense_attention_reference_(
        inputs.positions,
        inputs.q_normed,
        inputs.key_states,
        inputs.hidden_states,
        inputs.q_residual,
        inputs.query_start);
}

infinicore::Tensor DeepseekV4Attention::dense_attention_reference_(
    const std::vector<int64_t> &positions,
    const infinicore::Tensor &q_rope,
    const infinicore::Tensor &key_states,
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &q_residual,
    size_t query_start) const {
    const size_t compress_ratio = rotary_emb_.compress_ratio();
    if (compress_ratio == 0 && q_rope->device().getType() != infinicore::Device::Type::CPU) {
        return dense_attention_sliding_gpu_(q_rope, key_states, positions, query_start);
    }
    return dense_attention_compressed_cpu_(positions, q_rope, key_states, hidden_states, q_residual, query_start);
}

// -----------------------------------------------------------------------------
// Compressed attention (CSA / HCA) — CPU reference
// -----------------------------------------------------------------------------

infinicore::Tensor DeepseekV4Attention::dense_attention_compressed_cpu_(
    const std::vector<int64_t> &positions,
    const infinicore::Tensor &q_rope,
    const infinicore::Tensor &key_states,
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &q_residual,
    size_t query_start) const {
    const auto shape = q_rope->shape();
    const size_t batch_size = shape[0];
    const size_t query_len = shape[1];
    const size_t num_heads = shape[2];
    const size_t head_dim = shape[3];
    const size_t total_len = key_states->shape()[1];
    const size_t window = sliding_window_ == 0 ? total_len : sliding_window_;
    const size_t compress_ratio = rotary_emb_.compress_ratio();

    if (positions.size() < query_start + query_len) {
        throw std::runtime_error("DeepseekV4Attention: position_ids length mismatch");
    }

    auto q = tensor_to_float_vector(q_rope);
    auto kv = tensor_to_float_vector(key_states->contiguous());

    std::vector<float> kv_comp;
    size_t nb = 0;
    size_t index_top_k = 0;
    std::vector<int64_t> indexed_blocks;
    if (compressor_ && compress_ratio > 0) {
        size_t comp_batch = 0;
        kv_comp = compressor_->forward_values(hidden_states, comp_batch, nb);
        if (nb > 0 && indexer_) {
            indexed_blocks = indexer_->forward(
                hidden_states, q_residual, positions, index_top_k, query_start, query_len);
        }
    }

    auto sink = tensor_to_float_vector(attn_sink_);
    apply_compress_block_rope(
        kv_comp, batch_size, nb, head_dim, total_len, positions, compress_ratio, rotary_emb_.params());

    std::vector<float> out(batch_size * query_len * num_heads * head_dim, 0.0f);
    std::vector<float> logits(nb + total_len);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t tq = 0; tq < query_len; ++tq) {
            const size_t t = query_start + tq;
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t q_offset = ((b * query_len + tq) * num_heads + h) * head_dim;
                float max_logit = sink[h];

                for (size_t block = 0; block < nb; ++block) {
                    bool valid = static_cast<int64_t>(block)
                               < ((positions[t] + 1) / static_cast<int64_t>(compress_ratio));
                    if (valid && !indexed_blocks.empty()) {
                        valid = false;
                        const size_t index_offset = (b * query_len + tq) * index_top_k;
                        for (size_t k = 0; k < index_top_k; ++k) {
                            if (indexed_blocks[index_offset + k] == static_cast<int64_t>(block)) {
                                valid = true;
                                break;
                            }
                        }
                    }
                    if (!valid) {
                        logits[block] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    const size_t kv_offset = (b * nb + block) * head_dim;
                    double dot = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot += static_cast<double>(q[q_offset + d]) * kv_comp[kv_offset + d];
                    }
                    logits[block] = static_cast<float>(dot * softmax_scale_);
                    max_logit = std::max(max_logit, logits[block]);
                }

                for (size_t j = 0; j < total_len; ++j) {
                    const bool valid = positions[j] <= positions[t]
                                    && positions[j] > positions[t] - static_cast<int64_t>(window);
                    if (!valid) {
                        logits[nb + j] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    const size_t kv_offset = (b * total_len + j) * head_dim;
                    double dot = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) {
                        dot += static_cast<double>(q[q_offset + d]) * kv[kv_offset + d];
                    }
                    logits[nb + j] = static_cast<float>(dot * softmax_scale_);
                    max_logit = std::max(max_logit, logits[nb + j]);
                }

                double denom = std::exp(static_cast<double>(sink[h] - max_logit));
                for (float logit : logits) {
                    if (std::isfinite(logit)) {
                        denom += std::exp(static_cast<double>(logit - max_logit));
                    }
                }
                const size_t out_offset = ((b * query_len + tq) * num_heads + h) * head_dim;

                for (size_t block = 0; block < nb; ++block) {
                    if (!std::isfinite(logits[block])) {
                        continue;
                    }
                    const float prob = static_cast<float>(
                        std::exp(static_cast<double>(logits[block] - max_logit)) / denom);
                    const size_t kv_offset = (b * nb + block) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        out[out_offset + d] += prob * kv_comp[kv_offset + d];
                    }
                }
                for (size_t j = 0; j < total_len; ++j) {
                    if (!std::isfinite(logits[nb + j])) {
                        continue;
                    }
                    const float prob = static_cast<float>(
                        std::exp(static_cast<double>(logits[nb + j] - max_logit)) / denom);
                    const size_t kv_offset = (b * total_len + j) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        out[out_offset + d] += prob * kv[kv_offset + d];
                    }
                }

                apply_rope_at_offset(out, out_offset, positions[t], rotary_emb_.params(), true);
            }
        }
    }
    return float_vector_to_tensor(out, {batch_size, query_len, num_heads * head_dim}, q_rope->dtype(), q_rope->device());
}

// -----------------------------------------------------------------------------
// Sliding attention (compress_ratio == 0) — GPU QK^T + CPU sink softmax
// -----------------------------------------------------------------------------

infinicore::Tensor DeepseekV4Attention::dense_attention_sliding_gpu_(
    const infinicore::Tensor &q_rope,
    const infinicore::Tensor &key_states,
    const std::vector<int64_t> &positions,
    size_t query_start) const {
    const auto shape = q_rope->shape();
    const size_t batch_size = shape[0];
    const size_t query_len = shape[1];
    const size_t num_heads = shape[2];
    const size_t head_dim = shape[3];
    const size_t total_len = key_states->shape()[1];
    const size_t num_kv_heads = key_states->shape()[2];
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("DeepseekV4Attention: num_heads must be divisible by num_key_value_heads");
    }
    if (positions.size() < query_start + query_len) {
        throw std::runtime_error("DeepseekV4Attention: position_ids length mismatch");
    }
    const size_t ngroup = num_heads / num_kv_heads;
    const size_t window = sliding_window_ == 0 ? total_len : sliding_window_;

    auto q = q_rope->permute({0, 2, 1, 3})->contiguous();
    auto k = key_states->permute({0, 2, 1, 3})->contiguous();

    size_t kv_start = 0;
    size_t kv_len = total_len;
    if (sliding_window_ > 0 && query_len == 1) {
        const size_t t = query_start;
        const int64_t pos_min = positions[t] - static_cast<int64_t>(sliding_window_);
        while (kv_start < total_len && positions[kv_start] <= pos_min) {
            ++kv_start;
        }
        kv_len = total_len - kv_start;
        if (kv_len < total_len) {
            k = k->narrow({{2, kv_start, kv_len}})->contiguous();
        }
    }

    auto Q = q->view({batch_size * num_kv_heads, ngroup * query_len, head_dim});
    auto K = k->view({batch_size * num_kv_heads, kv_len, head_dim});
    auto scores = infinicore::op::matmul(Q, K->permute({0, 2, 1}), softmax_scale_);
    scores = scores->view({batch_size, num_heads, query_len, kv_len})->contiguous();

    if (cached_sink_host_.empty()) {
        cached_sink_host_ = tensor_to_float_vector(attn_sink_);
    }
    const auto &sink_host = cached_sink_host_;

    // InfiniCore add/cat/softmax on 4D BF16 attention scores can segfault; mask + sink softmax on CPU.
    auto scores_host = tensor_to_float_vector(scores);
    std::vector<float> probs_host(batch_size * num_heads * query_len * kv_len);
    std::vector<float> logits(kv_len + 1);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t tq = 0; tq < query_len; ++tq) {
            const size_t t = query_start + tq;
            std::vector<uint8_t> valid_keys(kv_len, 0);
            for (size_t j = 0; j < kv_len; ++j) {
                const size_t key_idx = kv_start + j;
                valid_keys[j] = positions[key_idx] <= positions[t]
                             && positions[key_idx] > positions[t] - static_cast<int64_t>(window);
            }
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t row_offset = ((b * num_heads + h) * query_len + tq) * kv_len;
                float max_logit = sink_host[h];
                for (size_t j = 0; j < kv_len; ++j) {
                    logits[j] = valid_keys[j] ? scores_host[row_offset + j] : -std::numeric_limits<float>::infinity();
                    if (std::isfinite(logits[j])) {
                        max_logit = std::max(max_logit, logits[j]);
                    }
                }
                logits[kv_len] = sink_host[h];
                max_logit = std::max(max_logit, logits[kv_len]);

                double denom = 0.0;
                for (float logit : logits) {
                    if (std::isfinite(logit)) {
                        denom += std::exp(static_cast<double>(logit - max_logit));
                    }
                }
                for (size_t j = 0; j < kv_len; ++j) {
                    if (!std::isfinite(logits[j])) {
                        probs_host[row_offset + j] = 0.0f;
                        continue;
                    }
                    probs_host[row_offset + j] = static_cast<float>(
                        std::exp(static_cast<double>(logits[j] - max_logit)) / denom);
                }
            }
        }
    }

    auto probs = float_vector_to_tensor(
        probs_host, {batch_size, num_heads, query_len, kv_len}, q_rope->dtype(), q_rope->device());
    auto probs_flat = probs->view({batch_size * num_kv_heads, ngroup * query_len, kv_len});
    auto V = k->view({batch_size * num_kv_heads, kv_len, head_dim});
    auto out = infinicore::op::matmul(probs_flat, V);
    out = out->view({batch_size, num_heads, query_len, head_dim})
              ->permute({0, 2, 1, 3})
              ->contiguous();

    std::vector<int64_t> query_positions(query_len);
    for (size_t tq = 0; tq < query_len; ++tq) {
        query_positions[tq] = positions[query_start + tq];
    }
    out = apply_rotary_pos_emb(out, query_positions, rotary_emb_.params(), true);
    return out->view({batch_size, query_len, num_heads * head_dim});
}

// -----------------------------------------------------------------------------
// Output projection (grouped wo_a → wo_b)
// -----------------------------------------------------------------------------

infinicore::Tensor DeepseekV4Attention::apply_grouped_output_projection_(const infinicore::Tensor &attn_output) const {
    const auto shape = attn_output->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto grouped = attn_output->view({batch_size * seq_len, o_groups_, o_a_input_size_});

    const auto wo_a_weight = wo_a_->weight();
    std::vector<infinicore::Tensor> projected_groups;
    projected_groups.reserve(o_groups_);
    for (size_t group_idx = 0; group_idx < o_groups_; ++group_idx) {
        auto group_input = grouped->narrow({{1, group_idx, 1}})->squeeze(1)->contiguous();
        auto group_weight = wo_a_weight->narrow({{0, group_idx * o_lora_rank_, o_lora_rank_}})->contiguous();
        auto group_output = infinicore::op::linear(group_input, group_weight, std::nullopt);
        projected_groups.push_back(group_output->view({batch_size, seq_len, o_lora_rank_}));
    }

    auto projected = infinicore::op::cat(projected_groups, 2);
    return wo_b_->forward(projected);
}

} // namespace infinilm::models::deepseek_v4
