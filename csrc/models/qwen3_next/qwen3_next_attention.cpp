#include "qwen3_next_attention.hpp"

#include "../../global_state/global_state.hpp"
#include <cmath>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::models::qwen3_next {

Qwen3NextAttention::Qwen3NextAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       size_t layer_idx,
                                       const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    const auto &dtype{model_config->get_dtype()};
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");

    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");

    bool use_bias = model_config->get_or<bool>("attention_bias", true);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    bool attn_output_gate = model_config->get_or<bool>("attn_output_gate", true);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    if ((total_num_kv_heads < tp_size) || (0 != (total_num_kv_heads % tp_size))) {
        throw std::runtime_error("infinilm::models::qwen3::Qwen3NextAttention: num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads / tp_size;

    auto quant_scheme = model_config->get_quant_scheme();
    auto quantization_method = model_config->get_quantization_method();
    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::NONE: {
        INFINILM_QKV_LINEAR_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, total_num_heads * (1 + attn_output_gate), total_num_kv_heads, quantization_method,
                                 use_bias, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method,
                                  use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        INFINILM_QKV_LINEAR_W8A8_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, total_num_heads * (1 + attn_output_gate), total_num_kv_heads, quantization_method,
                                      use_bias, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method,
                                  use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    case infinicore::quantization::QuantScheme::AWQ_W4A16: {
        INFINILM_QKV_LINEAR_W4A16AWQ_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, total_num_heads * (1 + attn_output_gate), total_num_kv_heads, quantization_method,
                                          use_bias, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method,
                                  use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    default: {
        throw std::runtime_error("infinilm::models::qwen3_next::Qwen3NextAttention: unsupported quantization scheme");
    }
    }

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
                                                                          kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);
}

infinicore::Tensor Qwen3NextAttention::forward(const infinicore::Tensor &positions,
                                               const infinicore::Tensor &hidden_states) const {
    spdlog::error("infinilm::models::qwen3_next::Qwen3NextAttention: forward not implemented");
    return hidden_states;
}

} // namespace infinilm::models::qwen3_next
