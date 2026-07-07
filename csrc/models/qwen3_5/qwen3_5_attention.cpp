#include "qwen3_5_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../layers/rotary_embedding/rotary_embedding_factory.hpp"
#include "../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <infinicore/ops/mul.hpp>
#include <infinicore/ops/sigmoid.hpp>
#include <optional>
#include <vector>

namespace infinilm::models::qwen3_5 {

Qwen35Attention::Qwen35Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 size_t layer_idx,
                                 const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");
    rotary_dim_ = model_config->get_rotary_dim();

    const auto &dtype{model_config->get_dtype()};
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("attention_bias", true);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();
    if ((total_num_kv_heads < tp_size) || (0 != (total_num_kv_heads % tp_size))) {
        throw std::runtime_error("infinilm::models::qwen3_5::Qwen35Attention: num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads / tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    qkv_proj_ = std::make_shared<Qwen35FusedQKVLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    const auto &rope_params = model_config->get_config_json()["rope_parameters"];
    const double partial_rotary_factor = rope_params["partial_rotary_factor"].get<double>();
    rotary_dim_ = static_cast<size_t>(std::llround(static_cast<double>(head_dim_) * partial_rotary_factor));
    rotary_dim_ = std::clamp(rotary_dim_, static_cast<size_t>(2), head_dim_);
    if (rotary_dim_ % 2 != 0) {
        rotary_dim_ -= 1;
    }
    rotary_dim_ = std::max(rotary_dim_, static_cast<size_t>(2));

    auto mrope_section = rope_params["mrope_section"].get<std::vector<int>>();
    if (mrope_section.size() != 3) {
        throw std::runtime_error("infinilm::models::qwen3_5::Qwen35Attention: mrope_section must have 3 elements");
    }
    const bool mrope_interleaved = rope_params["mrope_interleaved"].get<bool>();
    const double rope_theta = rope_params["rope_theta"].get<double>();
    const size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    auto rope_scaling = infinilm::layers::rotary_embedding::make_scaling_config(model_config);
    mrope_ = infinilm::layers::rotary_embedding::get_rope(head_dim_,
                                                          rotary_dim_,
                                                          max_position_embeddings,
                                                          rope_theta,
                                                          model_config->get_rope_algo(),
                                                          dtype,
                                                          device,
                                                          rope_scaling,
                                                          mrope_section,
                                                          mrope_interleaved);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
                                                                          kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);

    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor Qwen35Attention::forward(const infinicore::Tensor &positions,
                                            const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Qwen35Attention::forward_static_(const infinicore::Tensor &position_ids,
                                                    const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q, gate, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    auto q_reshaped = q_norm_->forward(q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
    auto k_reshaped = k_norm_->forward(k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));

    auto pos_shape = position_ids->shape();
    if (pos_shape.size() != 2 && pos_shape.size() != 1) {
        throw std::runtime_error("infinilm::models::qwen3_5::Qwen35Attention: Unexpected position_ids shape");
    }
    std::tie(q_reshaped, k_reshaped) = mrope_->forward(q_reshaped, k_reshaped, position_ids);

    q_reshaped = q_reshaped->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    k_reshaped = k_reshaped->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    attn_output = infinicore::op::mul(attn_output, infinicore::op::sigmoid(gate));
    return o_proj_->forward(attn_output);
}

infinicore::Tensor Qwen35Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                                   const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    ASSERT_EQ(batch_size, 1);

    auto [q, gate, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});
    q_reshaped = q_norm_->forward(q_reshaped);
    k_reshaped = k_norm_->forward(k_reshaped);

    auto pos_shape = position_ids->shape();
    if (pos_shape.size() != 2 && pos_shape.size() != 1) {
        throw std::runtime_error("Unexpected position_ids shape");
    }
    std::tie(q_reshaped, k_reshaped) = mrope_->forward(q_reshaped, k_reshaped, position_ids);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    attn_output = infinicore::op::mul(attn_output, infinicore::op::sigmoid(gate)->view(attn_output->shape()));
    return o_proj_->forward(attn_output);
}
} // namespace infinilm::models::qwen3_5
