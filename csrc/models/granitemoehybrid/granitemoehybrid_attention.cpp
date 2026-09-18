#include "granitemoehybrid_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../utils.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace infinilm::models::granitemoehybrid {

GraniteMoeHybridAttention::GraniteMoeHybridAttention(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx,
    const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get_head_dim();

    const auto &dtype{model_config->get_dtype()};
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    const bool use_bias = model_config->get_or<bool>("attention_bias", false);
    const bool use_output_bias =
        model_config->get_or<bool>("attention_output_bias", use_bias);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info =
        infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    if (tp_size <= 0 || total_num_heads % tp_size != 0) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridAttention: "
            "num_attention_heads must be divisible by tp_size");
    }
    if (total_num_kv_heads < static_cast<size_t>(tp_size) ||
        total_num_kv_heads % tp_size != 0) {
        throw std::runtime_error(
            "infinilm::models::granitemoehybrid::GraniteMoeHybridAttention: "
            "num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads / tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    
    qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
        hidden_size_,
        head_dim_,
        total_num_heads,
        total_num_kv_heads,
        "q_proj",
        "k_proj",
        "v_proj",
        register_fn,
        quantization_method,
        use_bias,
        dtype,
        device,
        rank_info);
    o_proj_ = this->register_module<infinilm::layers::linear::RowParallelLinear>(
        "o_proj",
        total_num_heads * head_dim_,
        hidden_size_,
        quantization_method,
        use_output_bias,
        dtype,
        device,
        tp_rank,
        tp_size,
        rank_info.comm);
    o_proj_->set_alpha(model_config->get_or<float>("residual_multiplier", 1.0f));

    const std::string position_embedding_type =
        model_config->get_or<std::string>("position_embedding_type", "nope");
    if ("rope" == position_embedding_type) {
        rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    }

    const float attention_multiplier =
        model_config->get_or<float>("attention_multiplier", 1.0f);
    infinilm::layers::attention::init_kv_cache_quant_params(
        register_fn,
        device,
        kv_cache_k_scale_,
        kv_cache_v_scale_);
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_,
        head_dim_,
        attention_multiplier,
        num_key_value_heads_,
        layer_idx_,
        kv_cache_k_scale_,
        kv_cache_v_scale_,
        attention_backend_);
}

infinicore::Tensor GraniteMoeHybridAttention::forward(
    const infinicore::Tensor &positions,
    const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor GraniteMoeHybridAttention::forward_static_(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    const auto &shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    auto [query, key, value] = qkv_proj_->forward_split(hidden_states_mutable);
    query = query->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    key = key->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    value = value->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    if (rotary_emb_) {
        const auto &position_shape = position_ids->shape();
        infinicore::Tensor rope_positions;
        if (position_shape.size() == 2) {
            rope_positions = position_ids->narrow({{0, 0, 1}})->contiguous()->view({position_shape[1]});
        } else if (position_shape.size() == 1) {
            rope_positions = position_ids->contiguous();
        } else {
            throw std::runtime_error(
                "infinilm::models::granitemoehybrid::GraniteMoeHybridAttention: "
                "unexpected position_ids shape");
        }

        rotary_emb_->forward(query, rope_positions, true);
        rotary_emb_->forward(key, rope_positions, true);
    }

    auto attention_output = attn_->forward(query, key, value);
    return o_proj_->forward(attention_output);
}

infinicore::Tensor GraniteMoeHybridAttention::forward_paged_(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    const auto &shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    ASSERT_EQ(batch_size, 1);

    auto [query, key, value] = qkv_proj_->forward_split(hidden_states_mutable);
    query = query->view({seq_len, num_attention_heads_, head_dim_});
    key = key->view({seq_len, num_key_value_heads_, head_dim_});
    value = value->view({seq_len, num_key_value_heads_, head_dim_});

    if (rotary_emb_) {
        const auto &position_shape = position_ids->shape();
        infinicore::Tensor rope_positions;
        if (position_shape.size() == 2) {
            rope_positions = position_ids->narrow({{0, 0, 1}})->view({position_shape[1]});
        } else if (position_shape.size() == 1) {
            rope_positions = position_ids;
        } else {
            throw std::runtime_error(
                "infinilm::models::granitemoehybrid::GraniteMoeHybridAttention: "
                "unexpected position_ids shape");
        }
        rotary_emb_->forward(query, rope_positions, true);
        rotary_emb_->forward(key, rope_positions, true);
    }

    auto attention_output = attn_->forward(query, key, value);
    return o_proj_->forward(attention_output);
}

} // namespace infinilm::models::granitemoehybrid
