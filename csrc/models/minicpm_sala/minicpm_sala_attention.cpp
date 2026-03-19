#include "minicpm_sala_attention.hpp"
#include "../../layers/common_modules.hpp"
#include <stdexcept>

namespace infinilm::models::minicpm_sala {

AttentionBase::AttentionBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t num_attention_heads,
                             size_t num_key_value_heads,
                             size_t layer_idx,
                             const infinicore::Device &device,
                             engine::distributed::RankInfo rank_info,
                             ::infinilm::backends::AttentionBackend attention_backend)
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

    scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    auto quant_scheme = this->model_config_->get_quant_scheme();
    auto quantization_method = this->model_config_->get_quantization_method();

    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::NONE:
        INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, num_attention_heads * head_dim_, quantization_method,
                                  use_output_bias_, dtype, device, tp_rank, tp_size);

        INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, num_key_value_heads * head_dim_, quantization_method,
                                  use_output_bias_, dtype, device, tp_rank, tp_size);

        INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, num_key_value_heads * head_dim_, quantization_method,
                                  use_output_bias_, dtype, device, tp_rank, tp_size);

        INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads * head_dim_, hidden_size_, quantization_method,
                                  use_output_bias_, dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    default:
        throw std::runtime_error("Unsupported quantization scheme");
        break;
    }

    if ((num_key_value_heads >= tp_size) && (0 == (num_key_value_heads % tp_size))) {
        this->num_attention_heads_ = num_attention_heads / tp_size;
        this->num_key_value_heads_ = num_key_value_heads / tp_size;
    } else {
        throw std::runtime_error("num_attention_heads / tp_size error.");
    }

    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling_, num_key_value_heads_, layer_idx_, attention_backend);
}

InfLLMv2Attention::InfLLMv2Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     size_t layer_idx,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info,
                                     ::infinilm::backends::AttentionBackend attention_backend)
    : AttentionBase(model_config,
                    model_config->get<size_t>("num_attention_heads"),
                    model_config->get<size_t>("num_key_value_heads"),
                    layer_idx, device, rank_info,
                    attention_backend),
      use_output_gate_(model_config->get_or<bool>("use_output_gate", false)) {

    const auto &dtype{model_config->get_dtype()};
    if (use_output_gate_) {
        INFINICORE_NN_MODULE_INIT(o_gate, hidden_size_, num_attention_heads_ * head_dim_,
                                  model_config->get_quantization_method(), use_bias_, dtype, device);
    }
}

infinicore::Tensor InfLLMv2Attention::forward(const infinicore::Tensor &hidden_states,
                                              const infinilm::InfinilmModel::Input &input,
                                              std::shared_ptr<infinilm::cache::Cache> kv_cache) const {
    spdlog::error("InfLLMv2Attention is not implemented");
    return hidden_states;
}

LightningAttention::LightningAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       size_t layer_idx,
                                       const infinicore::Device &device,
                                       engine::distributed::RankInfo rank_info,
                                       ::infinilm::backends::AttentionBackend attention_backend)
    : AttentionBase(model_config,
                    model_config->get<size_t>("num_attention_heads"),
                    model_config->get<size_t>("lightning_nkv"),
                    layer_idx, device, rank_info,
                    attention_backend),
      qk_norm_(model_config->get_or<bool>("qk_norm", false)),
      use_output_norm_(model_config->get_or<bool>("use_output_norm", false)),
      use_output_gate_(model_config->get_or<bool>("use_output_gate", false)) {

    const auto &dtype{model_config->get_dtype()};
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    if (qk_norm_) {
        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);
    }

    if (use_output_norm_) {
        INFINICORE_NN_MODULE_INIT(o_norm, num_attention_heads_ * head_dim_, rms_norm_eps, dtype, device);
    }

    if (use_output_gate_) {
        INFINICORE_NN_MODULE_INIT(z_proj, hidden_size_, num_attention_heads_ * head_dim_,
                                  model_config->get_quantization_method(), use_bias_, dtype, device);
    }
}

infinicore::Tensor LightningAttention::forward(const infinicore::Tensor &hidden_states,
                                               const infinilm::InfinilmModel::Input &input,
                                               std::shared_ptr<infinilm::cache::Cache> kv_cache) const {
    (void)input;
    (void)kv_cache;
    spdlog::error("LightningAttention is not implemented");
    return hidden_states;
}

} // namespace infinilm::models::minicpm_sala
