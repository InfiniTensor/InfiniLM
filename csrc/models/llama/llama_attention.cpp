#include "llama_attention.hpp"

#include "../../utils.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mul.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace infinilm::models::llama {

/**
 * @deprecated This function is deprecated and will be REMOVED in the next major release (v0.2.0).
 *
 * ⚠️ DEVELOPMENT POLICY:
 *   - NO new development or feature additions permitted on this interface
 *   - Only critical bug fixes (security/stability) allowed until removal
 *   - All new code MUST migrate to the polymorphic overload below
 *
 * Replacement: Use the polymorphic overload of this same function name with updated signature
 * Reason: Legacy signature lacks support for dynamic quantization modes.
 * Removal target: v0.2.0 (Q2 2026)
 */
LlamaAttention::LlamaAttention(const LlamaConfig &config,
                               const infinicore::Device &device,
                               size_t layer_idx,
                               engine::distributed::RankInfo rank_info)
    : layer_idx_(layer_idx),
      hidden_size_(config.hidden_size),
      num_attention_heads_(config.num_attention_heads),
      num_key_value_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      kv_dim_(config.kv_dim()),
      use_bias_(config.attention_bias),
      use_output_bias_(config.attention_output_bias),
      use_qk_norm_(config.qk_norm),
      max_position_embeddings_(config.max_position_embeddings), rank_info_(rank_info) {
    const auto &dtype{config.dtype};

    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    int num_attention_heads = config.num_attention_heads;
    int num_key_value_heads = config.num_key_value_heads;

    if ((num_key_value_heads >= tp_size) && (0 == (num_key_value_heads % tp_size))) {
        this->num_attention_heads_ = num_attention_heads / tp_size;
        this->num_key_value_heads_ = num_key_value_heads / tp_size;
    } else {
        throw std::runtime_error("num_attention_heads / tp_size error.");
    }
    scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // Initialize projection layers
    INFINILM_QKV_LINEAR_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, config.num_attention_heads, config.num_key_value_heads, use_bias_,
                             dtype, device, rank_info);
    // Output projection uses attention_output_bias (can be different from qkv)
    INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads * head_dim_, hidden_size_, use_output_bias_,
                              dtype, device, tp_rank, tp_size, rank_info.comm);

    // Initialize qk RMSNorm
    if (use_qk_norm_) {
        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, config.rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, config.rms_norm_eps, dtype, device);
    }
}

LlamaAttention::LlamaAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               const infinicore::Device &device,
                               size_t layer_idx,
                               engine::distributed::RankInfo rank_info)
    : model_config_(model_config),
      layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      num_attention_heads_(model_config->get<size_t>("num_attention_heads")),
      num_key_value_heads_(model_config->get<size_t>("num_key_value_heads")),
      head_dim_(model_config->get_head_dim()),
      kv_dim_(model_config->get_kv_dim()),
      use_bias_(model_config->get_or<bool>("attention_bias", true)),
      use_output_bias_(model_config->get_or<bool>("attention_output_bias", false)),
      max_position_embeddings_(model_config->get<size_t>("max_position_embeddings")),
      rank_info_(rank_info) {
    const auto &dtype{model_config_->get_dtype()};

    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    int num_attention_heads = model_config_->get<size_t>("num_attention_heads");
    int num_key_value_heads = model_config_->get<size_t>("num_key_value_heads");

    if ((num_key_value_heads >= tp_size) && (0 == (num_key_value_heads % tp_size))) {
        this->num_attention_heads_ = num_attention_heads / tp_size;
        this->num_key_value_heads_ = num_key_value_heads / tp_size;
    } else {
        throw std::runtime_error("num_attention_heads / tp_size error.");
    }
    scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    auto quant_scheme = this->model_config_->get_quant_scheme();
    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8:
        INFINILM_QKV_LINEAR_W8A8_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, model_config_->get<size_t>("num_attention_heads"), model_config_->get<size_t>("num_key_value_heads"), this->model_config_->get_quantization_method(), use_bias_,
                                      dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, model_config_->get<size_t>("num_attention_heads") * head_dim_, hidden_size_, this->model_config_->get_quantization_method(), use_output_bias_,
                                  dtype, device, tp_rank, tp_size, rank_info.comm);
        break;

    case infinicore::quantization::QuantScheme::AWQ_W4A16:
        INFINILM_QKV_LINEAR_W4A16AWQ_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, model_config_->get<size_t>("num_attention_heads"), model_config_->get<size_t>("num_key_value_heads"), this->model_config_->get_quantization_method(), use_bias_,
                                          dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, model_config_->get<size_t>("num_attention_heads") * head_dim_, hidden_size_, this->model_config_->get_quantization_method(), use_output_bias_,
                                  dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    default:
        INFINILM_QKV_LINEAR_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, model_config_->get<size_t>("num_attention_heads"), model_config_->get<size_t>("num_key_value_heads"), this->model_config_->get_quantization_method(), use_bias_,
                                 dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, model_config_->get<size_t>("num_attention_heads") * head_dim_, hidden_size_, this->model_config_->get_quantization_method(), use_output_bias_,
                                  dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    if (model_config_->get<std::string>("model_type") == "qwen3") {
        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, model_config_->get<double>("rms_norm_eps"), dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, model_config_->get<double>("rms_norm_eps"), dtype, device);
    }
}

infinicore::Tensor LlamaAttention::forward_(const infinicore::Tensor &hidden_states,
                                            const infinicore::Tensor &position_ids,
                                            std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                            std::optional<infinicore::Tensor> past_sequence_lengths,
                                            std::optional<infinicore::Tensor> total_sequence_lengths) const {
    // Input shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    if (use_qk_norm_ || model_config_->get_or<std::string>("model_type", "None") == "qwen3") {
        q = q_norm_->forward(q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
        k = k_norm_->forward(k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));
    }

    // 2. Reshape for multi-head attention
    // Reshape Q, K, V to include batch dimension
    // Python: query_states = self.q_proj(hidden_states).view(querys_shape)
    // The view operation requires the tensor to be contiguous in the required dimensions
    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE - align with Python pattern
    // Python: bs, num = pos_ids.shape; pos_ids = pos_ids.view((bs * num,))
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    // 4. Apply RoPE to Q and K
    auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
    rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope); // [bs, seq_len, n_q_head, head_dim]
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);   // [bs, seq_len, n_kv_head, head_dim]

    // 5. Prepare KV caches
    // Convert to [batch, n_head, seq_len, head_dim] for cache
    // Ensure contiguous after permute for F16 compatibility with cache operations
    q_reshaped = q_rope->permute({0, 2, 1, 3});          // [bs, n_q_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    infinicore::Tensor k_total;                          // [bs, n_kv_head, max_seq_len, head_dim]
    infinicore::Tensor v_total;                          // [bs, n_kv_head, max_seq_len, head_dim]
    if (kv_cache == nullptr) {
        k_total = k_permuted;
        v_total = v_permuted;
    } else if (auto static_kv_cache = std::dynamic_pointer_cast<cache::StaticKVCache>(kv_cache)) {
        auto [k_total_tmp, v_total_tmp] = static_kv_cache->update(layer_idx_, k_permuted, v_permuted, past_sequence_lengths.value());
        k_total = k_total_tmp;
        v_total = v_total_tmp;
    } else {
        throw std::runtime_error("LlamaAttention: Unsupported kvcache type");
    }

    infinicore::Tensor attn_output;
    if (false) {
        // experimental nineoothed flash attention
        attn_output = infinicore::op::flash_attention(q_reshaped, k_total, v_total, total_sequence_lengths.value(), scaling_, true);
        attn_output = attn_output->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_attention_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]
    } else {
        size_t total_seq_len = reinterpret_cast<int64_t *>(total_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];
        k_total = k_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]
        v_total = v_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]

        // 6. Compute attention
        size_t ngroup = num_attention_heads_ / num_key_value_heads_;
        auto Q = q_reshaped->view({batch_size * num_key_value_heads_, ngroup * seq_len, head_dim_});
        auto K = k_total->view({batch_size * num_key_value_heads_, total_seq_len, head_dim_});
        auto V = v_total->view({batch_size * num_key_value_heads_, total_seq_len, head_dim_});

        auto K_transposed = K->permute({0, 2, 1}); // [bs * n_kv_head, head_dim, total_seq_len]

        auto attn_weight = infinicore::op::matmul(Q, K_transposed, scaling_); // [bs * n_kv_head, ng * seq_len, total_seq_len]
        
        auto attn_weight_softmax = attn_weight->view({batch_size * num_attention_heads_, seq_len, total_seq_len});
        infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);

        auto out = infinicore::op::matmul(attn_weight, V); // [bs * n_kv_head, ng * seq_len, head_dim]

        attn_output = out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                          ->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_attention_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]
    }
    auto output = o_proj_->forward(attn_output);

    return output;
}

infinicore::Tensor LlamaAttention::forward_paged_(const infinicore::Tensor &hidden_states,
                                                  const infinicore::Tensor &position_ids,
                                                  std::shared_ptr<infinilm::cache::PagedKVCache> paged_kv_cache,
                                                  std::optional<infinicore::Tensor> total_sequence_lengths,
                                                  std::optional<infinicore::Tensor> input_offsets,
                                                  std::optional<infinicore::Tensor> block_tables,
                                                  std::optional<infinicore::Tensor> slot_mapping) const {
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());

    // Input shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // Only support batchsize==1, all requests should be flattened along seqlen dimension
    ASSERT_EQ(batch_size, 1);
    // Decode only if total_len == num_requests
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    // 2. Reshape for multi-head attention

    // Reshape Q, K, V to include batch dimension
    // Python: query_states = self.q_proj(hidden_states).view(querys_shape)
    // The view operation requires the tensor to be contiguous in the required dimensions
    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    if (use_qk_norm_ || model_config_->get_or<std::string>("model_type", "None") == "qwen3") {
        q_reshaped = q_norm_->forward(q_reshaped);
        k_reshaped = k_norm_->forward(k_reshaped);
    }

    // 3. Prepare position_ids for RoPE - align with Python pattern
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    // 4. Apply RoPE to Q and K
    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true); // [bs, seq_len, n_q_head, head_dim]
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true); // [bs, seq_len, n_kv_head, head_dim]

    //  5. Prepare KV caches
    //  Ensure contiguous after permute for F16 compatibility with cache operations
    auto [k_total, v_total] = paged_kv_cache->update(layer_idx_,
                                                     k_reshaped,
                                                     v_reshaped,
                                                     slot_mapping.value());

    // 6. Compute attention
    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_attention_heads_, head_dim_}, q_reshaped->dtype(), q_reshaped->device());

    if (is_prefill) {
        infinicore::op::paged_attention_prefill_(
            attn_output,
            q_reshaped,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            input_offsets.value(),
            std::nullopt,
            scaling_);

    } else {
        infinicore::op::paged_attention_(
            attn_output,
            q_reshaped,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            std::nullopt,
            scaling_);
    }

    // 7. Project output
    attn_output = attn_output->view({1, seq_len, num_attention_heads_ * head_dim_});
    return o_proj_->forward(attn_output);
}

infinicore::Tensor LlamaAttention::forward(const infinicore::Tensor &hidden_states,
                                           const infinicore::Tensor &position_ids,
                                           std::shared_ptr<cache::Cache> kv_cache,
                                           std::optional<infinicore::Tensor> past_sequence_lengths,
                                           std::optional<infinicore::Tensor> total_sequence_lengths,
                                           std::optional<infinicore::Tensor> input_offsets,
                                           std::optional<infinicore::Tensor> block_tables,
                                           std::optional<infinicore::Tensor> slot_mapping) const {
    if (!rotary_emb_) {
        throw std::runtime_error("LlamaAttention: rotary_emb not configured");
    }

    infinicore::Tensor output;
    if (auto paged_kv_cache = std::dynamic_pointer_cast<cache::PagedKVCache>(kv_cache)) {
        output = forward_paged_(hidden_states, position_ids, paged_kv_cache, total_sequence_lengths, input_offsets, block_tables, slot_mapping);
    } else {

        output = forward_(hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths);
    }
    return output;
}

void LlamaAttention::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    rotary_emb_ = rotary_emb;
}

} // namespace infinilm::models::llama
