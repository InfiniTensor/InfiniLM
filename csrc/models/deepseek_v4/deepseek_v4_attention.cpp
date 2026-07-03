#include "deepseek_v4_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "deepseek_v4_linear.hpp"
#include "deepseek_v4_utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/softmax.hpp"

#include <algorithm>
#include <cmath>
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

    auto q_normed = unweighted_rms_norm(q->contiguous(), rms_norm_eps_);
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
    const auto pos = normalize_positions(positions, seq_len);

    auto q_residual = q_norm_->forward(wq_a_->forward(hidden_states_mutable));

    auto q = wq_b_->forward(q_residual)->view({1, seq_len, num_attention_heads_, head_dim_});

    auto q_normed = unweighted_rms_norm(q->contiguous(), rms_norm_eps_);
    q_normed = rotary_emb_.forward(q_normed, pos);

    auto kv = kv_norm_->forward(wkv_->forward(hidden_states_mutable));

    auto key_states = rotary_emb_.forward(
        kv->view({1, seq_len, num_key_value_heads_, head_dim_}), pos);

    auto attn_output = dense_attention_reference_(positions, q_normed, key_states, hidden_states_mutable, q_residual);

    return apply_grouped_output_projection_(attn_output);
}

infinicore::Tensor DeepseekV4Attention::dense_attention_sliding_gpu_(const infinicore::Tensor &q_rope,
                                                                     const infinicore::Tensor &key_states,
                                                                     const std::vector<int64_t> &pos) const {
    const auto shape = q_rope->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    const size_t num_heads = shape[2];
    const size_t head_dim = shape[3];
    const size_t num_kv_heads = key_states->shape()[2];
    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error("DeepseekV4Attention: num_heads must be divisible by num_key_value_heads");
    }
    const size_t ngroup = num_heads / num_kv_heads;
    const size_t window = sliding_window_ == 0 ? seq_len : sliding_window_;

    auto q = q_rope->permute({0, 2, 1, 3})->contiguous();
    auto k = key_states->permute({0, 2, 1, 3})->contiguous();

    auto Q = q->view({batch_size * num_kv_heads, ngroup * seq_len, head_dim});
    auto K = k->view({batch_size * num_kv_heads, seq_len, head_dim});
    auto scores = infinicore::op::matmul(Q, K->permute({0, 2, 1}), softmax_scale_);
    scores = scores->view({batch_size, num_heads, seq_len, seq_len})->contiguous();

    // InfiniCore add/cat/softmax on 4D BF16 attention scores can segfault; mask + sink softmax on CPU.
    auto scores_host = tensor_to_float_vector(scores);
    const auto sink_host = tensor_to_float_vector(attn_sink_);
    std::vector<float> probs_host(batch_size * num_heads * seq_len * seq_len);
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t t = 0; t < seq_len; ++t) {
                const size_t row_offset = ((b * num_heads + h) * seq_len + t) * seq_len;
                float max_logit = sink_host[h];
                std::vector<float> logits(seq_len + 1);
                for (size_t j = 0; j < seq_len; ++j) {
                    const bool valid = pos[j] <= pos[t] && pos[j] > pos[t] - static_cast<int64_t>(window);
                    logits[j] = valid ? scores_host[row_offset + j] : -std::numeric_limits<float>::infinity();
                    if (std::isfinite(logits[j])) {
                        max_logit = std::max(max_logit, logits[j]);
                    }
                }
                logits[seq_len] = sink_host[h];
                max_logit = std::max(max_logit, logits[seq_len]);

                double denom = 0.0;
                for (float logit : logits) {
                    if (std::isfinite(logit)) {
                        denom += std::exp(static_cast<double>(logit - max_logit));
                    }
                }
                for (size_t j = 0; j < seq_len; ++j) {
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
        probs_host, {batch_size, num_heads, seq_len, seq_len}, q_rope->dtype(), q_rope->device());
    auto probs_flat = probs->view({batch_size * num_kv_heads, ngroup * seq_len, seq_len});
    auto V = k->view({batch_size * num_kv_heads, seq_len, head_dim});
    auto out = infinicore::op::matmul(probs_flat, V);
    out = out->view({batch_size, num_heads, seq_len, head_dim})
              ->permute({0, 2, 1, 3})
              ->contiguous();
    out = apply_rotary_pos_emb(out, pos, rotary_emb_.params(), true);
    return out->view({batch_size, seq_len, num_heads * head_dim});
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

    if (compress_ratio == 0
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
    // DeepseekV4HCACompressor，本处为 `Compressor.forward_values`）。每个块对 m 个源 token
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
        size_t comp_batch = 0; // 由 forward_values 写出；本单次 reference 路径未使用
        kv_comp = compressor_->forward_values(hidden_states, comp_batch, nb);
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
