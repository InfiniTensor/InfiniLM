#include "ernie4_5_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"

#include <cmath>

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5Attention::Ernie4_5Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");

    const auto &dtype = model_config->get_dtype();
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    const bool use_bias = model_config->get_or<bool>("attention_bias", false);
    const bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    const auto &config_json = model_config->get_config_json();

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    const int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads < static_cast<size_t>(tp_size)
                             ? 1
                             : total_num_kv_heads / tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);
    rope_theta_ = model_config->get_or<double>("rope_theta", 500000.0);
    if (config_json.contains("rope_scaling") && config_json["rope_scaling"].contains("mrope_section")) {
        const auto &sections = config_json["rope_scaling"]["mrope_section"];
        if (sections.is_array() && sections.size() == 3) {
            mrope_section_h_ = sections.at(0).get<size_t>();
            mrope_section_w_ = sections.at(1).get<size_t>();
            mrope_section_t_ = sections.at(2).get<size_t>();
        }
    }

    const float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor Ernie4_5Attention::forward(const infinicore::Tensor &positions,
                                              const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Ernie4_5Attention::forward_static_(const infinicore::Tensor &position_ids,
                                                      const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);
    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    auto pos_shape = position_ids->shape();
    if ((pos_shape.size() == 3 && pos_shape[0] == 1 && pos_shape[1] == seq_len && pos_shape[2] == 3) || (pos_shape.size() == 2 && pos_shape[0] == seq_len && pos_shape[1] == 3)) {
        ASSERT_EQ(batch_size, 1);
        auto q_mrope = q_reshaped->squeeze(0);
        auto k_mrope = k_reshaped->squeeze(0);
        infinicore::Tensor pos_ids_for_mrope = position_ids;
        if (pos_shape.size() == 3) {
            pos_ids_for_mrope = position_ids->squeeze(0);
        }
        infinicore::op::ernie45_mrope_(
            q_mrope, k_mrope, pos_ids_for_mrope,
            rope_theta_, mrope_section_h_, mrope_section_w_, mrope_section_t_);
        auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
        return o_proj_->forward(attn_output);
    }

    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("Ernie4_5Attention: Unexpected position_ids shape");
    }

    auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
    rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_rope, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

infinicore::Tensor Ernie4_5Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                                     const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);
    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    auto pos_shape = position_ids->shape();
    if ((pos_shape.size() == 3 && pos_shape[0] == 1 && pos_shape[1] == seq_len && pos_shape[2] == 3) || (pos_shape.size() == 2 && pos_shape[0] == seq_len && pos_shape[1] == 3)) {
        infinicore::Tensor pos_ids_for_mrope = position_ids;
        if (pos_shape.size() == 3) {
            pos_ids_for_mrope = position_ids->squeeze(0);
        }
        infinicore::op::ernie45_mrope_(
            q_reshaped, k_reshaped, pos_ids_for_mrope,
            rope_theta_, mrope_section_h_, mrope_section_w_, mrope_section_t_);
        auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
        return o_proj_->forward(attn_output);
    }

    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() != 1) {
        throw std::runtime_error("Ernie4_5Attention: Unexpected position_ids shape");
    }

    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

} // namespace infinilm::models::ernie4_5_moe_vl
