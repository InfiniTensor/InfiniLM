#include "qwen3_attention.hpp"

#include "../../config/infinilm_config.hpp"
#include "../../engine/parallel_state.hpp"
#include "../../utils.hpp"

namespace infinilm::models::qwen3 {

using infinilm::engine::AttentionMetadata;
using infinilm::engine::get_forward_context;

Qwen3Attention::Qwen3Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                               size_t layer_idx,
                               const infinicore::Device &device) {
    layer_idx_ = layer_idx;

    const auto &dtype{model_config->get_dtype()};
    num_attention_heads_ = model_config->get<size_t>("num_attention_heads");
    num_key_value_heads_ = model_config->get<size_t>("num_key_value_heads");
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    qk_norm_ = model_config->get_or<bool>("qk_norm", false);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    bool use_bias = model_config->get_or<bool>("attention_bias", true);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    auto quant_scheme = model_config->get_quant_scheme();
    auto quantization_method = model_config->get_quantization_method();

    attention_backend_ = infinilm::config::get_current_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::engine::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::engine::get_tensor_model_parallel_rank();
    int tp_size = infinilm::engine::get_tensor_model_parallel_world_size();

    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::NONE: {

        INFINILM_QKV_LINEAR_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, num_attention_heads_, num_key_value_heads_,
                                 quantization_method, use_bias, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads_ * head_dim_, hidden_size_, quantization_method,
                                  use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        INFINILM_QKV_LINEAR_W8A8_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, num_attention_heads_, num_key_value_heads_,
                                      quantization_method, use_bias, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads_ * head_dim_, hidden_size_, quantization_method,
                                  use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    case infinicore::quantization::QuantScheme::AWQ_W4A16: {
        INFINILM_QKV_LINEAR_W4A16AWQ_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, num_attention_heads_, num_key_value_heads_,
                                          quantization_method, use_bias, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads_ * head_dim_, hidden_size_, quantization_method,
                                  use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    default: {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: unsupported quantization scheme");
        break;
    }
    }

    if (qk_norm_) {
        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);
    }
    if ((num_key_value_heads_ < tp_size) || (0 != (num_key_value_heads_ % tp_size))) {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: num_key_value_heads must be divisible by tp_size");
    }

    size_t num_attention_heads_rank = num_attention_heads_ / tp_size;
    size_t num_key_value_heads_rank = num_key_value_heads_ / tp_size;
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_rank, head_dim_, scaling, num_key_value_heads_rank, layer_idx_, attention_backend_);
}

infinicore::Tensor Qwen3Attention::forward(const infinicore::Tensor &hidden_states) const {
    if (!rotary_emb_) {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: rotary_emb not configured");
    }

    auto &forward_context = get_forward_context();
    AttentionMetadata &attn_metadata = forward_context.attn_metadata;
    std::tuple<infinicore::Tensor, infinicore::Tensor> &kv_cache = forward_context.kv_cache_vec[layer_idx_];

    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(hidden_states, attn_metadata, kv_cache);
    }
    return forward_paged_(hidden_states, attn_metadata, kv_cache);
}

infinicore::Tensor Qwen3Attention::forward_static_(const infinicore::Tensor &hidden_states,
                                                   const infinilm::engine::AttentionMetadata &attn_metadata,
                                                   std::tuple<infinicore::Tensor, infinicore::Tensor> &kv_cache) const {

    auto position_ids = attn_metadata.position_ids.value();

    // hidden_states shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    if (qk_norm_) {
        q = q_norm_->forward(q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
        k = k_norm_->forward(k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));
    }

    // 2. Reshape for multi-head attention
    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3Attention: Unexpected position_ids shape");
    }

    // 4. Apply RoPE to QK
    auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
    rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    // 5. Prepare Attn
    // TODO: 确认这个shape对不对
    q_reshaped = q_rope->permute({0, 2, 1, 3});          // [bs, n_q_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]

    // 6. Attn Backend calculate
    auto attn_output = attn_->forward(q_reshaped, k_permuted, v_permuted, kv_cache, attn_metadata);

    // 7. Project output
    auto output = o_proj_->forward(attn_output);
    return output;
}

infinicore::Tensor Qwen3Attention::forward_paged_(const infinicore::Tensor &hidden_states,
                                                  const infinilm::engine::AttentionMetadata &attn_metadata,
                                                  std::tuple<infinicore::Tensor, infinicore::Tensor> &kv_cache) const {

    auto position_ids = attn_metadata.position_ids.value();

    // hidden_states shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // Only support batchsize==1, all requests should be flattened along seqlen dimension
    ASSERT_EQ(batch_size, 1);

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    // 2. Reshape for multi-head attention
    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    if (qk_norm_) {
        q_reshaped = q_norm_->forward(q_reshaped);
        k_reshaped = k_norm_->forward(k_reshaped);
    }

    // 3. Prepare position_ids for RoPE
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

    // 4. Apply RoPE to QK
    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    // 5. Attn Backend calculate
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, kv_cache, attn_metadata);

    // 6. Project output
    attn_output = attn_output->view({1, seq_len, num_attention_heads_ * head_dim_});
    return o_proj_->forward(attn_output);
}
} // namespace infinilm::models::qwen3
