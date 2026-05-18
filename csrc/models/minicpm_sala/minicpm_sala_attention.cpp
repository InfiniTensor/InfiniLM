#include "minicpm_sala_attention.hpp"
#include "../../global_state/global_state.hpp"
#include <stdexcept>

namespace infinilm::models::minicpm_sala {

AttentionBase::AttentionBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t num_attention_heads,
                             size_t num_key_value_heads,
                             size_t layer_idx,
                             const infinicore::Device &device)
    : layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      head_dim_(model_config->get<size_t>("head_dim")) {

    const auto &dtype{model_config->get_dtype()};

    use_bias_ = model_config->get_or<bool>("attention_bias", true);
    use_output_bias_ = model_config->get_or<bool>("attention_output_bias", false);

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    const size_t total_num_heads = num_attention_heads;
    const size_t total_num_kv_heads = num_key_value_heads;
    if ((total_num_kv_heads < static_cast<size_t>(tp_size)) || (0 != (total_num_kv_heads % static_cast<size_t>(tp_size)))) {
        throw std::runtime_error("infinilm::models::minicpm_sala::AttentionBase: num_key_value_heads must be divisible by tp_size");
    }

    num_attention_heads_ = total_num_heads / static_cast<size_t>(tp_size);
    num_key_value_heads_ = total_num_kv_heads / static_cast<size_t>(tp_size);

    auto quant_scheme = model_config->get_quant_scheme();
    auto quantization_method = model_config->get_quantization_method();
    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::NONE:
        INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, total_num_heads * head_dim_, quantization_method,
                                  use_bias_, dtype, device, tp_rank, tp_size);
        INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, total_num_kv_heads * head_dim_, quantization_method,
                                  use_bias_, dtype, device, tp_rank, tp_size);
        INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, total_num_kv_heads * head_dim_, quantization_method,
                                  use_bias_, dtype, device, tp_rank, tp_size);
        INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method,
                                  use_output_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    default:
        throw std::runtime_error("infinilm::models::minicpm_sala::AttentionBase: unsupported quantization scheme");
        break;
    }

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_, head_dim_, scaling,
                                                                          num_key_value_heads_, layer_idx_,
                                                                          kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    auto kv_quant_scheme = infinilm::global_state::get_infinilm_config().model_config->get_kv_quant_scheme();
    switch (kv_quant_scheme) {
    case (infinicore::quantization::KVQuantAlgo::NONE): {
        break;
    }
    case (infinicore::quantization::KVQuantAlgo::INT8): {
        INFINICORE_NN_PARAMETER_INIT(kv_cache_k_scale, ({1}, infinicore::DataType::F32, device, 0, 0, 1));
        INFINICORE_NN_PARAMETER_INIT(kv_cache_v_scale, ({1}, infinicore::DataType::F32, device, 0, 0, 1));
        break;
    }
    default: {
        throw std::runtime_error("infinilm::layers::attention: unsupported kv_quant_scheme");
        break;
    }
    }
}

InfLLMv2Attention::InfLLMv2Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device)
    : AttentionBase(model_config,
                    model_config->get<size_t>("num_attention_heads"),
                    model_config->get<size_t>("num_key_value_heads"),
                    layer_idx, device) {
    use_output_gate_ = model_config->get_or<bool>("use_output_gate", false);
    const auto &dtype{model_config->get_dtype()};
    size_t num_attention_heads = model_config->get<size_t>("num_attention_heads");
    if (use_output_gate_) {
        INFINICORE_NN_MODULE_INIT(o_gate, hidden_size_, num_attention_heads * head_dim_,
                                  model_config->get_quantization_method(), use_bias_, dtype, device);
    }
}

infinicore::Tensor InfLLMv2Attention::forward(const infinicore::Tensor &positions,
                                              const infinicore::Tensor &hidden_states) const {
    spdlog::error("InfLLMv2Attention is not implemented");
    return hidden_states;
}

LightningAttention::LightningAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       size_t layer_idx,
                                       const infinicore::Device &device)
    : AttentionBase(model_config,
                    model_config->get<size_t>("num_attention_heads"),
                    model_config->get<size_t>("lightning_nkv"),
                    layer_idx, device) {

    qk_norm_ = model_config->get_or<bool>("qk_norm", false);
    use_output_norm_ = model_config->get_or<bool>("use_output_norm", false);
    use_output_gate_ = model_config->get_or<bool>("use_output_gate", false);
    const auto &dtype{model_config->get_dtype()};
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    size_t num_attention_heads = model_config->get<size_t>("num_attention_heads");

    if (qk_norm_) {
        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);
    }
    if (use_output_norm_) {
        INFINICORE_NN_MODULE_INIT(o_norm, num_attention_heads * head_dim_, rms_norm_eps, dtype, device);
    }
    if (use_output_gate_) {
        INFINICORE_NN_MODULE_INIT(z_proj, hidden_size_, num_attention_heads * head_dim_,
                                  model_config->get_quantization_method(), use_bias_, dtype, device);
    }
}

infinicore::Tensor LightningAttention::forward(const infinicore::Tensor &positions,
                                               const infinicore::Tensor &hidden_states) const {
    spdlog::error("LightningAttention is not implemented");
    return hidden_states;
}

} // namespace infinilm::models::minicpm_sala
