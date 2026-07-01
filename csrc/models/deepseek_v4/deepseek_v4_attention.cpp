#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../utils.hpp"
#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/causal_softmax.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/matmul.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
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

std::vector<int64_t> normalize_positions(const infinicore::Tensor &positions, size_t seq_len) {
    auto values = tensor_to_int64_vector(positions);
    if (values.size() == seq_len) {
        return values;
    }
    if (values.size() >= seq_len) {
        return std::vector<int64_t>(values.end() - static_cast<std::ptrdiff_t>(seq_len), values.end());
    }
    std::vector<int64_t> out(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        out[i] = static_cast<int64_t>(i);
    }
    return out;
}

void apply_partial_rope_inplace(std::vector<float> &x,
                                size_t offset,
                                size_t head_dim,
                                size_t rope_dim,
                                int64_t position,
                                double theta,
                                bool inverse) {
    if (rope_dim == 0) {
        return;
    }
    const size_t pass_dim = head_dim - rope_dim;
    const size_t half = rope_dim / 2;
    std::vector<float> old(rope_dim);
    for (size_t i = 0; i < rope_dim; ++i) {
        old[i] = x[offset + pass_dim + i];
    }
    for (size_t i = 0; i < half; ++i) {
        const double inv_freq = 1.0 / std::pow(theta, static_cast<double>(2 * i) / static_cast<double>(rope_dim));
        const double angle = static_cast<double>(position) * inv_freq;
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>((inverse ? -1.0 : 1.0) * std::sin(angle));
        const float x1 = old[i];
        const float x2 = old[half + i];
        x[offset + pass_dim + i] = x1 * c - x2 * s;
        x[offset + pass_dim + half + i] = x2 * c + x1 * s;
    }
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
      qk_rope_head_dim_(model_config->get_or<size_t>("qk_rope_head_dim", 0)),
      sliding_window_(model_config->get_or<size_t>("sliding_window", 0)),
      rms_norm_eps_(model_config->get<double>("rms_norm_eps")),
      rope_theta_(model_config->get_or<double>("rope_theta", 10000.0)),
      compress_rope_theta_(model_config->get_or<double>("compress_rope_theta", model_config->get_or<double>("rope_theta", 10000.0))),
      softmax_scale_(1.0f / std::sqrt(static_cast<float>(head_dim_))) {
    const auto &dtype = model_config->get_dtype();
    const size_t q_output_size = num_attention_heads_ * head_dim_;
    size_t compress_ratio = 0;
    const auto &config_json = model_config->get_config_json();
    if (config_json.contains("compress_ratios") && layer_idx < config_json.at("compress_ratios").size()) {
        compress_ratio = config_json.at("compress_ratios").at(layer_idx).get<size_t>();
    }
    compress_ratio_ = compress_ratio;
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    auto quantization_method = deepseek_v4_linear_quantization(model_config, true);
    auto none_quantization = deepseek_v4_linear_quantization(model_config, false);
    INFINICORE_NN_MODULE_INIT(q_norm, q_lora_rank_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(kv_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(wq_a, hidden_size_, q_lora_rank_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wq_b, q_lora_rank_, q_output_size, quantization_method, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(wkv, hidden_size_, head_dim_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(wo_a, o_a_input_size_, o_a_output_size_, none_quantization, false, dtype, device, rank_info.tp_rank, rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(wo_b, o_a_output_size_, hidden_size_, quantization_method, false, dtype, device, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
    if (compress_ratio == 4) {
        INFINICORE_NN_MODULE_INIT(indexer, model_config, compress_ratio, device);
    }
    if (compress_ratio > 1) {
        INFINICORE_NN_MODULE_INIT(compressor, model_config, compress_ratio, head_dim_, device);
    }
    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

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

    auto q_residual = wq_a_->forward(hidden_states_mutable);
    q_residual = q_norm_->forward(q_residual);
    auto q = wq_b_->forward(q_residual)->view({batch_size, seq_len, num_attention_heads_, head_dim_});

    auto kv = wkv_->forward(hidden_states_mutable);
    kv = kv_norm_->forward(kv);
    auto key_states = kv->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto value_states = key_states;

    auto attn_output = dense_attention_reference_(positions, q, key_states, hidden_states_mutable, q_residual);
    return apply_grouped_output_projection_(attn_output);
}

infinicore::Tensor DeepseekV4Attention::forward_paged_(const infinicore::Tensor &positions,
                                                       const infinicore::Tensor &hidden_states) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);
    auto hidden_states_mutable = hidden_states;

    auto q_residual = wq_a_->forward(hidden_states_mutable);
    q_residual = q_norm_->forward(q_residual);
    auto q = wq_b_->forward(q_residual)->view({1, seq_len, num_attention_heads_, head_dim_});

    auto kv = wkv_->forward(hidden_states_mutable);
    kv = kv_norm_->forward(kv)->view({1, seq_len, num_key_value_heads_, head_dim_});
    auto key_states = kv;
    auto value_states = kv;

    auto attn_output = dense_attention_reference_(positions, q, key_states, hidden_states_mutable, q_residual);
    return apply_grouped_output_projection_(attn_output);
}

infinicore::Tensor DeepseekV4Attention::dense_attention_reference_(const infinicore::Tensor &positions,
                                                                   const infinicore::Tensor &query_states,
                                                                   const infinicore::Tensor &key_states,
                                                                   const infinicore::Tensor &hidden_states,
                                                                   const infinicore::Tensor &q_residual) const {
    const auto shape = query_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t num_heads = shape[2];
    const size_t head_dim = shape[3];
    const size_t window = sliding_window_ == 0 ? seq_len : sliding_window_;
    auto pos = normalize_positions(positions, seq_len);
    auto pos_ids = int64_vector_to_tensor(pos, {seq_len}, positions->device());
    auto q_normed = query_states->contiguous();
    {
        auto q_values = tensor_to_float_vector(q_normed);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t t = 0; t < seq_len; ++t) {
                for (size_t h = 0; h < num_heads; ++h) {
                    const size_t q_offset = ((b * seq_len + t) * num_heads + h) * head_dim;
                    double mean_square = 0.0;
                    for (size_t d = 0; d < head_dim; ++d) {
                        mean_square += static_cast<double>(q_values[q_offset + d]) * q_values[q_offset + d];
                    }
                    const float rsqrt = static_cast<float>(1.0 / std::sqrt(mean_square / static_cast<double>(head_dim) + rms_norm_eps_));
                    for (size_t d = 0; d < head_dim; ++d) {
                        q_values[q_offset + d] *= rsqrt;
                    }
                }
            }
        }
        q_normed = float_vector_to_tensor(q_values, {batch_size, seq_len, num_heads, head_dim}, query_states->dtype(), query_states->device());
    }
    auto q_rope = rotary_emb_->forward(
        q_normed->narrow({{3, head_dim - qk_rope_head_dim_, qk_rope_head_dim_}})->contiguous(),
        pos_ids,
        false);
    q_normed->narrow({{3, head_dim - qk_rope_head_dim_, qk_rope_head_dim_}})->copy_from(q_rope);
    auto kv_rope = key_states->contiguous();
    auto kv_rope_part = rotary_emb_->forward(
        kv_rope->narrow({{3, head_dim - qk_rope_head_dim_, qk_rope_head_dim_}})->contiguous(),
        pos_ids,
        false);
    kv_rope->narrow({{3, head_dim - qk_rope_head_dim_, qk_rope_head_dim_}})->copy_from(kv_rope_part);
    auto q = tensor_to_float_vector(q_normed);
    auto kv = tensor_to_float_vector(kv_rope);
    std::vector<float> kv_comp;
    size_t nb = 0;
    size_t index_top_k = 0;
    std::vector<int64_t> indexed_blocks;
    if (compressor_ && compress_ratio_ > 0) {
        size_t comp_batch = 0;
        kv_comp = compressor_->forward_values(hidden_states, comp_batch, nb);
        if (nb > 0 && indexer_) {
            indexed_blocks = indexer_->forward(hidden_states, q_residual, pos, index_top_k);
        }
    }
    auto sink = tensor_to_float_vector(attn_sink_);
    const int64_t max_position = pos.empty() ? 0 : *std::max_element(pos.begin(), pos.end());

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t block = 0; block < nb; ++block) {
            const int64_t block_pos = static_cast<int64_t>(block * compress_ratio_ + (compress_ratio_ - 1));
            const size_t kv_offset = (b * nb + block) * head_dim;
            apply_partial_rope_inplace(kv_comp, kv_offset, head_dim, qk_rope_head_dim_, std::min(block_pos, max_position), compress_rope_theta_, false);
        }
    }

    std::vector<float> out(batch_size * seq_len * num_heads * head_dim, 0.0f);
    std::vector<float> logits(nb + seq_len);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            for (size_t h = 0; h < num_heads; ++h) {
                const size_t q_offset = ((b * seq_len + t) * num_heads + h) * head_dim;
                float max_logit = sink[h];
                for (size_t block = 0; block < nb; ++block) {
                    bool valid = static_cast<int64_t>(block) < (pos[t] / static_cast<int64_t>(compress_ratio_));
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
                apply_partial_rope_inplace(out, out_offset, head_dim, qk_rope_head_dim_, pos[t], rope_theta_, true);
            }
        }
    }
    return float_vector_to_tensor(out, {batch_size, seq_len, num_heads * head_dim}, query_states->dtype(), query_states->device());
}

infinicore::Tensor DeepseekV4Attention::dense_attention_(const infinicore::Tensor &query_states,
                                                         const infinicore::Tensor &key_states,
                                                         const infinicore::Tensor &value_states) const {
    const auto shape = query_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t ngroup = num_attention_heads_ / num_key_value_heads_;

    auto q = query_states->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_key_value_heads_, ngroup * seq_len, head_dim_});
    auto k = key_states->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_key_value_heads_, seq_len, head_dim_});
    auto v = value_states->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_key_value_heads_, seq_len, head_dim_});
    auto k_t = k->permute({0, 2, 1});

    auto attn_weight = infinicore::op::matmul(q, k_t, softmax_scale_);
    auto attn_weight_softmax = attn_weight->view({batch_size * num_attention_heads_, seq_len, seq_len});
    infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);

    auto out = infinicore::op::matmul(attn_weight, v);
    return out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
        ->permute({0, 2, 1, 3})
        ->contiguous()
        ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
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
    return wo_b_->forward(projected);
}

} // namespace infinilm::models::deepseek_v4
