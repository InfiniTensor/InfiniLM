#include "minicpm_sala_attention.hpp"
#include <stdexcept>

namespace infinilm::models::minicpm_sala {

AttentionBase::AttentionBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t num_attention_heads,
                             size_t num_key_value_heads,
                             const infinicore::Device &device,
                             size_t layer_idx,
                             engine::distributed::RankInfo rank_info)
    : model_config_(model_config),
      layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      num_attention_heads_(num_attention_heads),
      num_key_value_heads_(num_key_value_heads),
      head_dim_(model_config->get_head_dim()),
      kv_dim_(model_config->get_kv_dim()),
      use_bias_(model_config->get_or<bool>("attention_bias", true)),
      use_output_bias_(model_config->get_or<bool>("attention_output_bias", false)),
      max_position_embeddings_(model_config->get<size_t>("max_position_embeddings")),
      rank_info_(rank_info) {
    const auto &dtype{model_config_->get_dtype()};
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;

    if ((num_key_value_heads >= tp_size) && (0 == (num_key_value_heads % tp_size))) {
        this->num_attention_heads_ = num_attention_heads / tp_size;
        this->num_key_value_heads_ = num_key_value_heads / tp_size;
    } else {
        throw std::runtime_error("num_attention_heads / tp_size error.");
    }

    scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    auto quant_scheme = this->model_config_->get_quant_scheme();
    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::NONE:

        INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, num_attention_heads * head_dim_, this->model_config_->get_quantization_method(), use_output_bias_,
                                  dtype, device, tp_rank, tp_size);

        INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, num_key_value_heads * head_dim_, this->model_config_->get_quantization_method(), use_output_bias_,
                                  dtype, device, tp_rank, tp_size);

        INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, num_key_value_heads * head_dim_, this->model_config_->get_quantization_method(), use_output_bias_,
                                  dtype, device, tp_rank, tp_size);

        INFINICORE_NN_MODULE_INIT(o_proj, model_config_->get<size_t>("num_attention_heads") * head_dim_, hidden_size_, this->model_config_->get_quantization_method(), use_output_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    default:
        throw std::runtime_error("Unsupported quantization scheme");
        break;
    }
}

InfLLMv2Attention::InfLLMv2Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device,
                                     size_t layer_idx,
                                     engine::distributed::RankInfo rank_info)
    : AttentionBase(model_config,
                    model_config->get<size_t>("num_attention_heads"),
                    model_config->get<size_t>("num_key_value_heads"),
                    device, layer_idx, rank_info),
      use_output_gate_(model_config->get_or<bool>("use_output_gate", false)) {

    const auto &dtype{model_config_->get_dtype()};
    if (use_output_gate_) {
        INFINICORE_NN_MODULE_INIT(o_gate, hidden_size_, num_attention_heads_ * head_dim_, model_config_->get_quantization_method(), use_bias_, dtype, device);
    }
}

infinicore::Tensor InfLLMv2Attention::forward(const infinicore::Tensor &hidden_states,
                                              const infinilm::InfinilmModel::Input &input,
                                              std::shared_ptr<infinilm::cache::Cache> kv_cache) const {
    throw std::runtime_error("InfLLMv2Attention is not supported");
    return hidden_states;
}

LightningAttention::LightningAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       const infinicore::Device &device,
                                       size_t layer_idx,
                                       engine::distributed::RankInfo rank_info)
    : AttentionBase(model_config,
                    model_config->get<size_t>("num_attention_heads"),
                    model_config->get<size_t>("lightning_nkv"),
                    device, layer_idx, rank_info),
      use_qk_norm_(model_config_->get_use_qk_norm()),
      use_output_norm_(model_config->get_or<bool>("use_output_norm", false)),
      use_output_gate_(model_config->get_or<bool>("use_output_gate", false)) {

    const auto &dtype{model_config_->get_dtype()};

    if (use_qk_norm_) {
        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, model_config_->get<double>("rms_norm_eps"), dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, model_config_->get<double>("rms_norm_eps"), dtype, device);
    }

    if (use_output_norm_) {
        INFINICORE_NN_MODULE_INIT(o_norm, num_attention_heads_ * head_dim_, model_config_->get<double>("rms_norm_eps"), dtype, device);
    }

    if (use_output_gate_) {
        INFINICORE_NN_MODULE_INIT(z_proj, hidden_size_, num_attention_heads_ * head_dim_, model_config_->get_quantization_method(), use_bias_, dtype, device);
    }
}

infinicore::Tensor LightningAttention::forward(const infinicore::Tensor &hidden_states,
                                               const infinilm::InfinilmModel::Input &input,
                                               std::shared_ptr<infinilm::cache::Cache> kv_cache) const {
    throw std::runtime_error("LightningAttention is not supported");
    return hidden_states;
}

} // namespace infinilm::models::minicpm_sala
