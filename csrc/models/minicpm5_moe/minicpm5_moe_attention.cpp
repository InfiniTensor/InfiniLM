#include "minicpm5_moe_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/sigmoid.hpp>
#include <stdexcept>

namespace infinilm::models::minicpm5_moe {

MiniCPM5MoEAttention::MiniCPM5MoEAttention(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    use_gated_attention_ = model_config->get_or<bool>("use_gated_attention", true);

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

    // Gated path doubles q_proj (per-head [q | gate]), same packing as qwen3_next.
    const size_t q_heads_for_proj = total_num_heads * (use_gated_attention_ ? 2 : 1);

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) {
        this->register_parameter(n, std::move(p));
    };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, q_heads_for_proj, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    infinilm::layers::attention::init_kv_cache_quant_params(
        register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor MiniCPM5MoEAttention::forward(
    const infinicore::Tensor &positions,
    const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor MiniCPM5MoEAttention::forward_static_(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q_proj_out, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    infinicore::Tensor q_reshaped;
    infinicore::Tensor gate;
    if (use_gated_attention_) {
        auto q_proj_heads = q_proj_out->as_strided(
            {batch_size * seq_len, num_attention_heads_, head_dim_ * 2},
            {q_proj_out->stride(1), static_cast<infinicore::Stride>(head_dim_ * 2), 1});
        auto q = q_proj_heads->narrow({{2, 0, head_dim_}});
        gate = q_proj_heads->narrow({{2, head_dim_, head_dim_}})
                   ->contiguous()
                   ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
        q_reshaped = q->as_strided(
            {batch_size, seq_len, num_attention_heads_, head_dim_},
            {static_cast<infinicore::Stride>(seq_len * num_attention_heads_ * head_dim_),
             static_cast<infinicore::Stride>(num_attention_heads_ * head_dim_),
             static_cast<infinicore::Stride>(head_dim_),
             1});
    } else {
        q_reshaped = q_proj_out->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    }

    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error(
            "MiniCPM5MoEAttention: Unexpected position_ids shape");
    }

    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    if (use_gated_attention_) {
        attn_output = infinicore::op::mul(attn_output, infinicore::op::sigmoid(gate));
    }
    return o_proj_->forward(attn_output);
}

infinicore::Tensor MiniCPM5MoEAttention::forward_paged_(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    auto [q_proj_out, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    infinicore::Tensor q_reshaped;
    infinicore::Tensor gate;
    if (use_gated_attention_) {
        auto q_proj_heads = q_proj_out->as_strided(
            {seq_len, num_attention_heads_, head_dim_ * 2},
            {q_proj_out->stride(1), static_cast<infinicore::Stride>(head_dim_ * 2), 1});
        auto q = q_proj_heads->narrow({{2, 0, head_dim_}});
        gate = q_proj_heads->narrow({{2, head_dim_, head_dim_}})
                   ->contiguous()
                   ->view({seq_len, num_attention_heads_ * head_dim_});
        q_reshaped = q;
    } else {
        q_reshaped = q_proj_out->view({seq_len, num_attention_heads_, head_dim_});
    }

    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("MiniCPM5MoEAttention: Unexpected position_ids shape");
    }

    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    if (use_gated_attention_) {
        attn_output = infinicore::op::mul(
            attn_output, infinicore::op::sigmoid(gate)->view(attn_output->shape()));
    }
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::minicpm5_moe
