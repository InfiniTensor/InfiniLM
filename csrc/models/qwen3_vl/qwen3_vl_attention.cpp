#include "qwen3_vl_attention.hpp"

#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../layers/rotary_embedding/rotary_embedding_factory.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace infinilm::models::qwen3_vl {

Qwen3VLAttention::Qwen3VLAttention(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    size_t layer_idx, const infinicore::Device &device)
    : layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      head_dim_(model_config->get<size_t>("head_dim")) {
    const auto &dtype = model_config->get_dtype();
    const size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    const size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    const bool use_bias = model_config->get_or<bool>("attention_bias", true);
    const bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    attention_backend_ = global_state::get_infinilm_config().attention_backend;
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    if (total_num_heads % static_cast<size_t>(tp_size) != 0) {
        throw std::runtime_error(
            "Qwen3VLAttention: num_attention_heads must be divisible by tp_size");
    }
    if (total_num_kv_heads >= static_cast<size_t>(tp_size) && total_num_kv_heads % static_cast<size_t>(tp_size) != 0) {
        throw std::runtime_error(
            "Qwen3VLAttention: num_key_value_heads must be divisible by tp_size");
    }
    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
    num_key_value_heads_ = total_num_kv_heads < static_cast<size_t>(tp_size)
                             ? 1
                             : total_num_kv_heads / static_cast<size_t>(tp_size);

    auto quantization = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &name,
                              infinicore::nn::Parameter parameter) {
        this->register_parameter(name, std::move(parameter));
    };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads, "q_proj",
        "k_proj", "v_proj", register_fn, quantization, use_bias, dtype, device,
        rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    const auto &rope_scaling_json = model_config->get_config_json().at("rope_scaling");
    const auto mrope_section = rope_scaling_json.at("mrope_section").get<std::vector<int>>();
    if (mrope_section.size() != 3) {
        throw std::runtime_error(
            "Qwen3VLAttention: mrope_section must have three entries");
    }
    const bool mrope_interleaved = rope_scaling_json.at("mrope_interleaved").get<bool>();
    auto scaling = layers::rotary_embedding::make_scaling_config(model_config);
    mrope_ = layers::rotary_embedding::get_rope(
        head_dim_, head_dim_,
        model_config->get<size_t>("max_position_embeddings"),
        model_config->get<double>("rope_theta"), model_config->get_rope_algo(),
        dtype, device, scaling, mrope_section, mrope_interleaved);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scale, num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);
    INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);
    layers::attention::init_kv_cache_quant_params(
        register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor
Qwen3VLAttention::forward(const infinicore::Tensor &positions,
                          const infinicore::Tensor &hidden_states) const {
    if (attention_backend_ == backends::AttentionBackend::STATIC_ATTN) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Qwen3VLAttention::forward_static_(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states) const {
    auto hidden_mutable = hidden_states;
    const size_t batch_size = hidden_states->size(0);
    const size_t seq_len = hidden_states->size(1);
    auto [q, k, v] = qkv_proj_->forward_split(hidden_mutable);

    q = q_norm_->forward(
        q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
    k = k_norm_->forward(
        k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));
    std::tie(q, k) = mrope_->forward(q, k, position_ids);

    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    output = o_proj_->forward(output);
    return output;
}

infinicore::Tensor Qwen3VLAttention::forward_paged_(
    const infinicore::Tensor &position_ids,
    const infinicore::Tensor &hidden_states) const {
    ASSERT_EQ(hidden_states->size(0), 1);
    auto hidden_mutable = hidden_states;
    const size_t seq_len = hidden_states->size(1);
    auto [q, k, v] = qkv_proj_->forward_split(hidden_mutable);

    q = q_norm_->forward(q->view({seq_len, num_attention_heads_, head_dim_}));
    k = k_norm_->forward(k->view({seq_len, num_key_value_heads_, head_dim_}));
    std::tie(q, k) = mrope_->forward(q, k, position_ids);

    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});
    auto output = attn_->forward(q, k, v_reshaped);
    return o_proj_->forward(output);
}

} // namespace infinilm::models::qwen3_vl
