#include "minimax_text_01_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <stdexcept>

namespace infinilm::models::minimax_text_01 {

MiniMaxText01Attention::MiniMaxText01Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                               size_t layer_idx,
                                               const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");

    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");

    bool use_bias = model_config->get_or<bool>("attention_bias", false);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads < static_cast<size_t>(tp_size)
                             ? 1
                             : total_num_kv_heads / tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
                                                                          kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor MiniMaxText01Attention::forward(const infinicore::Tensor &positions,
                                                   const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor MiniMaxText01Attention::forward_static_(const infinicore::Tensor &position_ids,
                                                           const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Fused QKV projection, then slice into q / k / v.
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    // 2. Reshape to heads. MiniMax full attention has no QK normalization.
    auto q_heads = q->as_strided(
        {batch_size * seq_len, num_attention_heads_, head_dim_},
        {q->stride(1), static_cast<infinicore::Stride>(head_dim_), 1});
    auto k_heads = k->as_strided(
        {batch_size * seq_len, num_key_value_heads_, head_dim_},
        {k->stride(1), static_cast<infinicore::Stride>(head_dim_), 1});

    // 3. Reshape to [batch, seq, heads, head_dim] for the attention kernel.
    auto q_reshaped = q_heads->as_strided(
        {batch_size, seq_len, num_attention_heads_, head_dim_},
        {static_cast<infinicore::Stride>(seq_len * num_attention_heads_ * head_dim_),
         static_cast<infinicore::Stride>(num_attention_heads_ * head_dim_),
         static_cast<infinicore::Stride>(head_dim_),
         1});
    auto k_reshaped = k_heads->as_strided(
        {batch_size, seq_len, num_key_value_heads_, head_dim_},
        {static_cast<infinicore::Stride>(seq_len * num_key_value_heads_ * head_dim_),
         static_cast<infinicore::Stride>(num_key_value_heads_ * head_dim_),
         static_cast<infinicore::Stride>(head_dim_),
         1});
    auto v_reshaped = v->as_strided(
        {batch_size, seq_len, num_key_value_heads_, head_dim_},
        {v->stride(0), v->stride(1), static_cast<infinicore::Stride>(head_dim_), 1});

    // 4. Prepare position ids for RoPE.
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("infinilm::models::minimax_text_01::MiniMaxText01Attention: Unexpected position_ids shape");
    }

    // 5. Apply RoPE to Q and K.
    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    // 6. Attention kernel (updates the KV cache internally).
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);

    // 7. Output projection.
    return o_proj_->forward(attn_output);
}

infinicore::Tensor MiniMaxText01Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                                          const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    // 1. Fused QKV projection, then slice into q / k / v.
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    // 2. Reshape to heads and apply QK normalization (flattened sequence layout).
    auto q_heads = q->as_strided(
        {seq_len, num_attention_heads_, head_dim_},
        {q->stride(1), static_cast<infinicore::Stride>(head_dim_), 1});
    auto q_reshaped = q_heads;
    auto k_heads = k->as_strided(
        {seq_len, num_key_value_heads_, head_dim_},
        {k->stride(1), static_cast<infinicore::Stride>(head_dim_), 1});
    auto k_reshaped = k_heads;
    auto v_reshaped = v->as_strided(
        {seq_len, num_key_value_heads_, head_dim_},
        {v->stride(1), static_cast<infinicore::Stride>(head_dim_), 1});

    // 3. Prepare position ids for RoPE.
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("infinilm::models::minimax_text_01::MiniMaxText01Attention: Unexpected position_ids shape");
    }

    // 4. Apply RoPE to Q and K.
    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    // 5. Attention kernel (updates the paged KV cache internally).
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);

    // 6. Output projection.
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::minimax_text_01
