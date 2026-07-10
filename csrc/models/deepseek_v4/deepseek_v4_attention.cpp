#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/deepseek_v4_compressed_decode.hpp"
#include "infinicore/ops/deepseek_v4_swa_decode.hpp"
#include "infinicore/ops/deepseek_v4_swa_prefill.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/softmax.hpp"
#include "infinicore/ops/unweighted_rms_norm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace infinilm::models::deepseek_v4 {
namespace {

void warn_attention_approximation_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        spdlog::warn("DeepseekV4Attention uses a reference CPU path for V4 sliding/compressed attention.");
    });
}

bool disable_fused_swa_decode() {
    static const bool value = std::getenv("DSV4_DISABLE_FUSED_SWA_DECODE") != nullptr;
    return value;
}

bool disable_fused_swa_prefill() {
    static const bool value = std::getenv("DSV4_DISABLE_FUSED_SWA_PREFILL") != nullptr;
    return value;
}

size_t fused_swa_prefill_min_len() {
    static const size_t value = []() {
        const char *env = std::getenv("DSV4_FUSED_SWA_PREFILL_MIN_LEN");
        if (!env || env[0] == '\0') {
            return static_cast<size_t>(128);
        }
        return static_cast<size_t>(std::max(1, std::atoi(env)));
    }();
    return value;
}

bool disable_compressed_empty_fastpath() {
    static const bool value = std::getenv("DSV4_DISABLE_COMPRESSED_EMPTY_FASTPATH") != nullptr;
    return value;
}

bool disable_compressed_decode() {
    static const bool value = std::getenv("DSV4_DISABLE_COMPRESSED_DECODE") != nullptr;
    return value;
}

bool disable_compressed_kv_cache() {
    static const bool value = std::getenv("DSV4_DISABLE_COMPRESSED_KV_CACHE") != nullptr;
    return value;
}

bool disable_decode_position_fastpath() {
    static const bool value = std::getenv("DSV4_DISABLE_DECODE_POSITION_FASTPATH") != nullptr;
    return value;
}

bool use_cpu_compressor_reference() {
    static const bool value = std::getenv("DSV4_COMPRESSOR_CPU") != nullptr;
    return value;
}

bool has_no_visible_compressed_blocks(size_t compress_ratio,
                                      const std::vector<int64_t> &positions,
                                      size_t query_start,
                                      size_t query_len) {
    if (compress_ratio == 0) {
        return true;
    }
    if (disable_compressed_empty_fastpath() || query_len == 0
        || positions.size() < query_start + query_len) {
        return false;
    }
    const int64_t ratio = static_cast<int64_t>(compress_ratio);
    for (size_t tq = 0; tq < query_len; ++tq) {
        const int64_t visible_blocks = (positions[query_start + tq] + 1) / ratio;
        if (visible_blocks > 0) {
            return false;
        }
    }
    return true;
}

} // namespace

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
    auto hidden_states_mutable = hidden_states;
    const auto pos = normalize_positions(positions, seq_len);

    auto q_residual = wq_a_->forward(hidden_states_mutable);
    q_residual = q_norm_->forward(q_residual);

    auto q = wq_b_->forward(q_residual)->view({batch_size, seq_len, num_attention_heads_, head_dim_});

    auto q_normed = infinicore::op::unweighted_rms_norm(q->contiguous(), static_cast<float>(rms_norm_eps_));
    q_normed = rotary_emb_.forward(q_normed, pos);

    auto kv = wkv_->forward(hidden_states_mutable);

    kv = kv_norm_->forward(kv);

    auto key_states = rotary_emb_.forward(
        kv->view({batch_size, seq_len, num_key_value_heads_, head_dim_}), pos);

    auto attn_output = dense_attention_reference_(positions, q_normed, key_states, hidden_states_mutable, q_residual);

    return apply_grouped_output_projection_(attn_output);
}

infinicore::Tensor DeepseekV4Attention::forward_paged_(const infinicore::Tensor &positions,
                                                       const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);
    auto hidden_states_mutable = hidden_states;
    std::vector<int64_t> pos;
    const bool decode_position_fastpath = !disable_decode_position_fastpath()
                                      && cached_seq_len_ > 0 && seq_len == 1;
    if (decode_position_fastpath) {
        pos = {static_cast<int64_t>(cached_seq_len_)};
    } else {
        pos = normalize_positions(positions, seq_len);
    }

    auto q_residual = q_norm_->forward(wq_a_->forward(hidden_states_mutable));

    auto q = wq_b_->forward(q_residual)->view({1, seq_len, num_attention_heads_, head_dim_});

    auto q_normed = infinicore::op::unweighted_rms_norm(q->contiguous(), static_cast<float>(rms_norm_eps_));
    q_normed = rotary_emb_.forward(q_normed, pos);

    auto kv = kv_norm_->forward(wkv_->forward(hidden_states_mutable));

    auto key_states = rotary_emb_.forward(
        kv->view({1, seq_len, num_key_value_heads_, head_dim_}), pos);

    const bool is_decode = cached_seq_len_ > 0 && seq_len == 1 && !pos.empty()
                        && pos[0] >= static_cast<int64_t>(cached_seq_len_);

    infinicore::Tensor attn_output;
    if (is_decode) {
        const size_t query_start = cached_seq_len_;
        append_decode_cache_(hidden_states_mutable, q_residual, key_states, pos);
        if (has_no_visible_compressed_blocks(rotary_emb_.compress_ratio(),
                                             cached_positions_,
                                             query_start,
                                             seq_len)
            && q_normed->device().getType() != infinicore::Device::Type::CPU) {
            attn_output = dense_attention_sliding_gpu_(
                q_normed, cached_key_states_, cached_positions_, query_start);
        } else {
            attn_output = dense_attention_decode_reference_(
                q_normed, cached_key_states_, cached_hidden_states_, cached_q_residual_, cached_positions_, query_start);
        }
    } else {
        cached_hidden_states_ = hidden_states_mutable;
        cached_q_residual_ = q_residual;
        cached_key_states_ = key_states;
        cached_positions_ = pos;
        cached_seq_len_ = seq_len;
        cached_hidden_states_storage_.reset();
        cached_q_residual_storage_.reset();
        cached_key_states_storage_.reset();
        cached_storage_capacity_ = seq_len;
        cached_kv_comp_tensor_.reset();
        cached_kv_comp_blocks_ = 0;
        cached_kv_comp_batch_ = 0;
        cached_block_positions_tensor_.reset();
        cached_block_positions_blocks_ = 0;
        attn_output = dense_attention_reference_(positions, q_normed, key_states, hidden_states_mutable, q_residual);
    }

    return apply_grouped_output_projection_(attn_output);
}

void DeepseekV4Attention::append_decode_cache_(const infinicore::Tensor &hidden_states,
                                               const infinicore::Tensor &q_residual,
                                               const infinicore::Tensor &key_states,
                                               const std::vector<int64_t> &positions) const {
    const size_t old_len = cached_seq_len_;
    const size_t append_len = hidden_states->shape()[1];
    if (append_len == 0) {
        return;
    }
    const size_t new_len = old_len + append_len;

    if (!cached_hidden_states_ || !cached_q_residual_ || !cached_key_states_) {
        cached_hidden_states_ = hidden_states;
        cached_q_residual_ = q_residual;
        cached_key_states_ = key_states;
        cached_positions_.insert(cached_positions_.end(), positions.begin(), positions.end());
        cached_seq_len_ = cached_positions_.size();
        cached_storage_capacity_ = cached_seq_len_;
        return;
    }

    if (cached_storage_capacity_ < new_len || !cached_hidden_states_storage_
        || !cached_q_residual_storage_ || !cached_key_states_storage_) {
        size_t new_capacity = cached_storage_capacity_ == 0 ? new_len : cached_storage_capacity_ * 2;
        new_capacity = std::max(new_capacity, old_len + static_cast<size_t>(16));
        new_capacity = std::max(new_capacity, new_len);

        auto grow_storage = [&](const infinicore::Tensor &current, infinicore::Tensor &storage) {
            auto shape = current->shape();
            shape[1] = new_capacity;
            auto next_storage = infinicore::Tensor::empty(shape, current->dtype(), current->device());
            if (old_len > 0) {
                auto dst = next_storage->narrow({{1, 0, old_len}});
                dst->copy_from(current);
            }
            storage = next_storage;
        };

        grow_storage(cached_hidden_states_, cached_hidden_states_storage_);
        grow_storage(cached_q_residual_, cached_q_residual_storage_);
        grow_storage(cached_key_states_, cached_key_states_storage_);
        cached_storage_capacity_ = new_capacity;
    }

    auto append_to_storage = [&](infinicore::Tensor &view,
                                 const infinicore::Tensor &storage,
                                 const infinicore::Tensor &value) {
        auto dst = storage->narrow({{1, old_len, append_len}});
        dst->copy_from(value);
        view = storage->narrow({{1, 0, new_len}});
    };

    append_to_storage(cached_hidden_states_, cached_hidden_states_storage_, hidden_states);
    append_to_storage(cached_q_residual_, cached_q_residual_storage_, q_residual);
    append_to_storage(cached_key_states_, cached_key_states_storage_, key_states);
    cached_positions_.insert(cached_positions_.end(), positions.begin(), positions.end());
    cached_seq_len_ = cached_positions_.size();
}

infinicore::Tensor DeepseekV4Attention::dense_attention_decode_reference_(const infinicore::Tensor &q_rope,
                                                                          const infinicore::Tensor &key_states,
                                                                          const infinicore::Tensor &hidden_states,
                                                                          const infinicore::Tensor &q_residual,
                                                                          const std::vector<int64_t> &positions,
                                                                          size_t query_start) const {
    const auto q_shape = q_rope->shape();
    const size_t batch_size = q_shape[0];
    const size_t query_len = q_shape[1];
    const size_t num_heads = q_shape[2];
    const size_t head_dim = q_shape[3];
    const size_t total_len = key_states->shape()[1];
    const size_t window = sliding_window_ == 0 ? total_len : sliding_window_;
    const size_t compress_ratio = rotary_emb_.compress_ratio();

    std::vector<float> kv_comp;
    infinicore::Tensor kv_comp_tensor;
    size_t comp_batch = 0;
    size_t nb = 0;
    size_t index_top_k = 0;
    infinicore::Tensor indexed_blocks_tensor;
    std::vector<int64_t> indexed_blocks;
    if (compressor_ && compress_ratio > 0) {
        const size_t expected_nb = total_len / compress_ratio;
        if (expected_nb == 0) {
            comp_batch = batch_size;
            nb = 0;
        } else if (!use_cpu_compressor_reference()
                   && hidden_states->device().getType() != infinicore::Device::Type::CPU) {
            if (!disable_compressed_kv_cache()
                && cached_kv_comp_tensor_ && cached_kv_comp_batch_ == batch_size
                && cached_kv_comp_blocks_ == expected_nb) {
                kv_comp_tensor = cached_kv_comp_tensor_;
                comp_batch = cached_kv_comp_batch_;
                nb = cached_kv_comp_blocks_;
            } else {
                kv_comp_tensor = compressor_->forward_tensor(hidden_states, comp_batch, nb);
                if (!disable_compressed_kv_cache()) {
                    cached_kv_comp_tensor_ = kv_comp_tensor;
                    cached_kv_comp_batch_ = comp_batch;
                    cached_kv_comp_blocks_ = nb;
                }
            }
        } else {
            kv_comp = compressor_->forward_values(hidden_states, comp_batch, nb);
        }
        if (nb > 0 && indexer_) {
            indexed_blocks_tensor = indexer_->forward_tensor(hidden_states, q_residual, positions, index_top_k,
                                                             query_start, query_len);
        }
    }

    if (!disable_compressed_decode() && query_len == 1 && kv_comp_tensor && nb > 0
        && nb + (sliding_window_ == 0 ? total_len : std::min(total_len, sliding_window_)) <= 4096
        && q_rope->device().getType() != infinicore::Device::Type::CPU) {
        size_t kv_start = 0;
        size_t kv_len = total_len;
        if (sliding_window_ > 0) {
            const int64_t pos_min = positions[query_start] - static_cast<int64_t>(sliding_window_);
            while (kv_start < total_len && positions[kv_start] <= pos_min) {
                ++kv_start;
            }
            kv_len = total_len - kv_start;
        }
        auto k_window = key_states;
        if (kv_start > 0 || kv_len < total_len) {
            k_window = key_states->narrow({{1, kv_start, kv_len}});
        }

        std::vector<int64_t> query_positions(query_len);
        for (size_t tq = 0; tq < query_len; ++tq) {
            query_positions[tq] = positions[query_start + tq];
        }
        const auto &rope_params = rotary_emb_.params();
        auto query_positions_tensor = int64_vector_to_tensor(
            query_positions, {query_len}, q_rope->device());
        infinicore::Tensor block_positions_tensor;
        if (cached_block_positions_tensor_ && cached_block_positions_blocks_ == nb) {
            block_positions_tensor = cached_block_positions_tensor_;
        } else {
            std::vector<int64_t> block_positions(nb);
            for (size_t block = 0; block < nb; ++block) {
                const size_t block_token = std::min(block * compress_ratio, total_len > 0 ? total_len - 1 : 0);
                block_positions[block] = (positions[block_token] / static_cast<int64_t>(compress_ratio))
                                       * static_cast<int64_t>(compress_ratio);
            }
            block_positions_tensor = int64_vector_to_tensor(
                block_positions, {nb}, q_rope->device());
            cached_block_positions_tensor_ = block_positions_tensor;
            cached_block_positions_blocks_ = nb;
        }
        size_t gpu_index_top_k = 0;
        auto gpu_indexed_blocks_tensor = int64_vector_to_tensor(
            std::vector<int64_t>{-1}, {1}, q_rope->device());
        if (indexed_blocks_tensor && index_top_k > 0) {
            gpu_index_top_k = index_top_k;
            gpu_indexed_blocks_tensor = indexed_blocks_tensor->view({indexed_blocks_tensor->numel()})->contiguous();
        }
        auto out = infinicore::op::deepseek_v4_compressed_decode(
            q_rope->contiguous(),
            key_states->contiguous(),
            kv_comp_tensor->contiguous(),
            attn_sink_->contiguous(),
            query_positions_tensor,
            block_positions_tensor,
            gpu_indexed_blocks_tensor,
            kv_start,
            kv_len,
            softmax_scale_,
            compress_ratio,
            gpu_index_top_k,
            rope_params.rope_dim,
            rope_params.rope_theta,
            rope_params.use_yarn,
            rope_params.yarn_factor,
            rope_params.yarn_beta_fast,
            rope_params.yarn_beta_slow,
            rope_params.yarn_original_seq_len,
            rope_params.yarn_extrapolation_factor);
        return out->view({batch_size, query_len, num_heads * head_dim});
    }

    auto q = tensor_to_float_vector(q_rope);
    auto kv = tensor_to_float_vector(key_states->contiguous());
    if (kv_comp_tensor && kv_comp.empty()) {
        kv_comp = tensor_to_float_vector(kv_comp_tensor);
    }
    rotary_emb_.forward_blocks(kv_comp, batch_size, nb, head_dim, total_len, positions);

    auto sink = tensor_to_float_vector(attn_sink_);
    std::vector<float> out(batch_size * query_len * num_heads * head_dim, 0.0f);
    std::vector<float> logits(nb + total_len);

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t tq = 0; tq < query_len; ++tq) {
            const size_t t = query_start + tq;
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t q_offset = ((b * query_len + tq) * num_heads + h) * head_dim;
                float max_logit = sink[h];

                for (size_t block = 0; block < nb; ++block) {
                    bool valid = static_cast<int64_t>(block) < ((positions[t] + 1) / static_cast<int64_t>(compress_ratio));
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
                    const bool valid = positions[j] <= positions[t] && positions[j] > positions[t] - static_cast<int64_t>(window);
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
                    const float prob = static_cast<float>(std::exp(static_cast<double>(logits[block] - max_logit)) / denom);
                    const size_t kv_offset = (b * nb + block) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        out[out_offset + d] += prob * kv_comp[kv_offset + d];
                    }
                }
                for (size_t j = 0; j < total_len; ++j) {
                    if (!std::isfinite(logits[nb + j])) {
                        continue;
                    }
                    const float prob = static_cast<float>(std::exp(static_cast<double>(logits[nb + j] - max_logit)) / denom);
                    const size_t kv_offset = (b * total_len + j) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        out[out_offset + d] += prob * kv[kv_offset + d];
                    }
                }

                rotary_emb_.inverse_at_offset(out, out_offset, positions[t]);
            }
        }
    }

    return float_vector_to_tensor(out, {batch_size, query_len, num_heads * head_dim}, q_rope->dtype(), q_rope->device());
}

infinicore::Tensor DeepseekV4Attention::dense_attention_sliding_gpu_(const infinicore::Tensor &q_rope,
                                                                     const infinicore::Tensor &key_states,
                                                                     const std::vector<int64_t> &pos,
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
    if (pos.size() < query_start + query_len) {
        throw std::runtime_error("DeepseekV4Attention: position_ids length mismatch");
    }
    const size_t ngroup = num_heads / num_kv_heads;
    const size_t window = sliding_window_ == 0 ? total_len : sliding_window_;

    if (!disable_fused_swa_prefill() && query_len >= fused_swa_prefill_min_len()
        && q_rope->device().getType() != infinicore::Device::Type::CPU) {
        std::vector<int64_t> query_positions(query_len);
        for (size_t tq = 0; tq < query_len; ++tq) {
            query_positions[tq] = pos[query_start + tq];
        }
        auto query_positions_tensor = int64_vector_to_tensor(
            query_positions, {query_len}, q_rope->device());
        auto key_positions_tensor = int64_vector_to_tensor(
            pos, {total_len}, q_rope->device());
        const auto &rope_params = rotary_emb_.params();
        auto out = infinicore::op::deepseek_v4_swa_prefill(
            q_rope->contiguous(),
            key_states->contiguous(),
            attn_sink_->contiguous(),
            query_positions_tensor,
            key_positions_tensor,
            softmax_scale_,
            window,
            rope_params.rope_dim,
            rope_params.rope_theta,
            rope_params.use_yarn,
            rope_params.yarn_factor,
            rope_params.yarn_beta_fast,
            rope_params.yarn_beta_slow,
            rope_params.yarn_original_seq_len,
            rope_params.yarn_extrapolation_factor);
        return out->view({batch_size, query_len, num_heads * head_dim});
    }

    size_t kv_start = 0;
    size_t kv_len = total_len;
    if (sliding_window_ > 0 && query_len == 1) {
        const size_t t = query_start;
        const int64_t pos_min = pos[t] - static_cast<int64_t>(sliding_window_);
        while (kv_start < total_len && pos[kv_start] <= pos_min) {
            ++kv_start;
        }
        kv_len = total_len - kv_start;
    }

    if (!disable_fused_swa_decode() && query_len == 1) {
        auto k_window = key_states;
        if (kv_start > 0 || kv_len < total_len) {
            k_window = key_states->narrow({{1, kv_start, kv_len}});
        }
        std::vector<int64_t> query_positions(query_len);
        for (size_t tq = 0; tq < query_len; ++tq) {
            query_positions[tq] = pos[query_start + tq];
        }
        const auto &rope_params = rotary_emb_.params();
        auto positions_tensor = int64_vector_to_tensor(
            query_positions, {query_len}, q_rope->device());
        auto out = infinicore::op::deepseek_v4_swa_decode(
            q_rope->contiguous(),
            key_states->contiguous(),
            attn_sink_->contiguous(),
            positions_tensor,
            kv_start,
            kv_len,
            softmax_scale_,
            rope_params.rope_dim,
            rope_params.rope_theta,
            rope_params.use_yarn,
            rope_params.yarn_factor,
            rope_params.yarn_beta_fast,
            rope_params.yarn_beta_slow,
            rope_params.yarn_original_seq_len,
            rope_params.yarn_extrapolation_factor);
        return out->view({batch_size, query_len, num_heads * head_dim});
    }

    auto q = q_rope->permute({0, 2, 1, 3})->contiguous();
    auto k = key_states->permute({0, 2, 1, 3})->contiguous();
    if (kv_len < total_len) {
        k = k->narrow({{2, kv_start, kv_len}})->contiguous();
    }

    auto Q = q->view({batch_size * num_kv_heads, ngroup * query_len, head_dim});
    auto K = k->view({batch_size * num_kv_heads, kv_len, head_dim});
    auto scores = infinicore::op::matmul(Q, K->permute({0, 2, 1}), softmax_scale_);
    scores = scores->view({batch_size, num_heads, query_len, kv_len})->contiguous();

    // InfiniCore add/cat/softmax on 4D BF16 attention scores can segfault; mask + sink softmax on CPU.
    auto scores_host = tensor_to_float_vector(scores);
    const auto sink_host = tensor_to_float_vector(attn_sink_);
    std::vector<float> probs_host(batch_size * num_heads * query_len * kv_len);
    std::vector<float> logits(kv_len + 1);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t tq = 0; tq < query_len; ++tq) {
            const size_t t = query_start + tq;
            std::vector<uint8_t> valid_keys(kv_len, 0);
            for (size_t j = 0; j < kv_len; ++j) {
                const size_t key_idx = kv_start + j;
                valid_keys[j] = pos[key_idx] <= pos[t]
                             && pos[key_idx] > pos[t] - static_cast<int64_t>(window);
            }
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t row_offset = ((b * num_heads + h) * query_len + tq) * kv_len;
                float max_logit = sink_host[h];
                for (size_t j = 0; j < kv_len; ++j) {
                    logits[j] = valid_keys[j] ? scores_host[row_offset + j]
                                              : -std::numeric_limits<float>::infinity();
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
        query_positions[tq] = pos[query_start + tq];
    }
    out = apply_rotary_pos_emb(out, query_positions, rotary_emb_.params(), true);
    return out->view({batch_size, query_len, num_heads * head_dim});
}

infinicore::Tensor DeepseekV4Attention::dense_attention_reference_(const infinicore::Tensor &positions,
                                                                   const infinicore::Tensor &q_rope,
                                                                   const infinicore::Tensor &key_states,
                                                                   const infinicore::Tensor &hidden_states,
                                                                   const infinicore::Tensor &q_residual) const {
    const auto shape = q_rope->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t num_heads = shape[2];
    const size_t head_dim = shape[3];
    const size_t window = sliding_window_ == 0 ? seq_len : sliding_window_;
    const size_t compress_ratio = rotary_emb_.compress_ratio();
    auto pos = normalize_positions(positions, seq_len);

    if (has_no_visible_compressed_blocks(compress_ratio, pos, 0, seq_len)
        && q_rope->device().getType() != infinicore::Device::Type::CPU) {
        return dense_attention_sliding_gpu_(q_rope, key_states, pos);
    }

    auto q = tensor_to_float_vector(q_rope);
    auto kv = tensor_to_float_vector(key_states->contiguous());

    // --- 构建 attention KV 轴上的压缩分支（仅 CSA / HCA 层）---
    //
    // DeepSeek-V4 的 attention 在两类 key 上做注意力（见 modeling_deepseek_v4.py 的
    // `torch.cat([kv, compressed_kv], dim=2)`，以及 sglang 的 `Compressor` / `C4Indexer`）：
    //   1. 滑动窗口 token：来自 `kv_proj` 的逐位置 `kv`（上文已做 RoPE）。
    //   2. 压缩块：compressor 将每 m 个 token 聚合成一个向量。
    //
    // | 层类型 (compress_ratios[layer]) | compressor_ | indexer_ | 行为 |
    // |--------------------------------|-------------|----------|------|
    // | 0  (sliding_attention)         | 无          | 无       | 仅 SW |
    // | 4  (compressed_sparse_attention) | CSA m=4   | Lightning| top-k 块 + SW |
    // | 128 (heavily_compressed_attention) | HCA m=128 | 无    | 全部块 + SW |
    //
    // kv_comp：展平为 [B * nb * head_dim] 的压缩 KV（对应 HF 的 DeepseekV4CSACompressor /
    // DeepseekV4HCACompressor，本处优先使用 GPU compressor，必要时回退 `Compressor.forward_values`）。每个块对 m 个源 token
    // 做 gated softmax 池化（CSA 为 Ca/Cb 重叠布局，coff_=2）。
    // nb：序列中完整 m-token 窗口个数（usable_len / m）。
    //
    // indexed_blocks + index_top_k：仅 CSA 层的稀疏筛选（HF `DeepseekV4Indexer`，
    // sglang `C4Indexer`）。对每个 query 位置 t，indexer 用
    // sum_h w_{t,h} * ReLU(q_{t,h} · K^IComp_s) 给压缩块打分，保留 top index_topk 索引。
    // 无效/未来块记为 -1。下游（约 217–228 行）块可见当且仅当
    // block < (pos[t]+1)/m 且（无 indexer 或 block ∈ indexed_blocks[t, :k]）。
    // HCA 层无 indexer，对所有因果压缩块做 attention。
    std::vector<float> kv_comp;
    size_t nb = 0;
    size_t index_top_k = 0;
    std::vector<int64_t> indexed_blocks;
    if (compressor_ && compress_ratio > 0) {
        size_t comp_batch = 0;
        if (!use_cpu_compressor_reference()
            && hidden_states->device().getType() != infinicore::Device::Type::CPU) {
            auto kv_comp_tensor = compressor_->forward_tensor(hidden_states, comp_batch, nb);
            kv_comp = tensor_to_float_vector(kv_comp_tensor);
        } else {
            kv_comp = compressor_->forward_values(hidden_states, comp_batch, nb);
        }
        if (nb > 0 && indexer_) {
            indexed_blocks = indexer_->forward(hidden_states, q_residual, pos, index_top_k);
        }
    }

    auto sink = tensor_to_float_vector(attn_sink_);

    rotary_emb_.forward_blocks(kv_comp, batch_size, nb, head_dim, seq_len, pos);

    std::vector<float> out(batch_size * seq_len * num_heads * head_dim, 0.0f);
    std::vector<float> logits(nb + seq_len);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t q_offset = ((b * seq_len + t) * num_heads + h) * head_dim;
                float max_logit = sink[h];

                for (size_t block = 0; block < nb; ++block) {
                    bool valid = static_cast<int64_t>(block) < ((pos[t] + 1) / static_cast<int64_t>(compress_ratio));
                    if (valid && !indexed_blocks.empty()) {
                        valid = false;
                        const size_t index_offset = (b * seq_len + t) * index_top_k;
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

                for (size_t j = 0; j < seq_len; ++j) {
                    const bool valid = pos[j] <= pos[t] && pos[j] > pos[t] - static_cast<int64_t>(window);
                    if (!valid) {
                        logits[nb + j] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    const size_t kv_offset = (b * seq_len + j) * head_dim;
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
                const size_t out_offset = ((b * seq_len + t) * num_heads + h) * head_dim;

                for (size_t block = 0; block < nb; ++block) {
                    if (!std::isfinite(logits[block])) {
                        continue;
                    }
                    const float prob = static_cast<float>(std::exp(static_cast<double>(logits[block] - max_logit)) / denom);
                    const size_t kv_offset = (b * nb + block) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        out[out_offset + d] += prob * kv_comp[kv_offset + d];
                    }
                }
                for (size_t j = 0; j < seq_len; ++j) {
                    if (!std::isfinite(logits[nb + j])) {
                        continue;
                    }
                    const float prob = static_cast<float>(std::exp(static_cast<double>(logits[nb + j] - max_logit)) / denom);
                    const size_t kv_offset = (b * seq_len + j) * head_dim;
                    for (size_t d = 0; d < head_dim; ++d) {
                        out[out_offset + d] += prob * kv[kv_offset + d];
                    }
                }

                rotary_emb_.inverse_at_offset(out, out_offset, pos[t]);
            }
        }
    }
    return float_vector_to_tensor(out, {batch_size, seq_len, num_heads * head_dim}, q_rope->dtype(), q_rope->device());
}

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

    auto final_output = wo_b_->forward(projected);

    return final_output;
}

} // namespace infinilm::models::deepseek_v4
