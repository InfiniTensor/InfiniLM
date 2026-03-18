#include "attn_base.hpp"

#include "../../utils.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/mul.hpp"

#include "infinicore/io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace infinilm::models::layers::attention {
AttentionBase::AttentionBase(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device,
                             size_t layer_idx,
                             engine::distributed::RankInfo rank_info)
    : model_config_(model_config),
      layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      head_dim_(model_config->get_head_dim()),
      kv_dim_(model_config->get_kv_dim()),
      use_bias_(model_config->get_or<bool>("attention_bias", true)),
      qk_norm_(model_config->get_or<bool>("qk_norm", false)),
      use_output_bias_(model_config->get_or<bool>("attention_output_bias", false)),
      max_position_embeddings_(model_config->get<size_t>("max_position_embeddings")),
      rank_info_(rank_info) {

    const auto &dtype{model_config_->get_dtype()};
    int tp_rank = rank_info.tp_rank;
    int tp_size = rank_info.tp_size;
    int num_attention_heads = model_config_->get<size_t>("num_attention_heads");
    int num_key_value_heads = model_config_->get<size_t>("num_key_value_heads");
    double rms_norm_eps = model_config_->get<double>("rms_norm_eps");

    if ((num_key_value_heads >= tp_size) && (0 == (num_key_value_heads % tp_size))) {
        this->num_attention_heads_ = num_attention_heads / tp_size;
        this->num_key_value_heads_ = num_key_value_heads / tp_size;
    } else {
        throw std::runtime_error("num_attention_heads / tp_size error.");
    }
    scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    auto quant_scheme = this->model_config_->get_quant_scheme();
    switch (quant_scheme) {
    case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8:
        INFINILM_QKV_LINEAR_W8A8_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, num_attention_heads, num_key_value_heads, this->model_config_->get_quantization_method(), use_bias_,
                                      dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads * head_dim_, hidden_size_, this->model_config_->get_quantization_method(), use_output_bias_,
                                  dtype, device, tp_rank, tp_size, rank_info.comm);
        break;

    case infinicore::quantization::QuantScheme::AWQ_W4A16: {
        INFINILM_QKV_LINEAR_W4A16AWQ_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, num_attention_heads, num_key_value_heads, this->model_config_->get_quantization_method(), use_bias_, dtype, device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads * head_dim_, hidden_size_, this->model_config_->get_quantization_method(), use_output_bias_,
                                  dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }
    default:
        INFINILM_QKV_LINEAR_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, num_attention_heads, num_key_value_heads, this->model_config_->get_quantization_method(), use_bias_,
                                 dtype,
                                 device, rank_info);
        INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads * head_dim_, hidden_size_, this->model_config_->get_quantization_method(), use_output_bias_,
                                  dtype, device, tp_rank, tp_size, rank_info.comm);
        break;
    }

    if (qk_norm_) {
        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);
    }
}
} // namespace infinilm::models::layers::attention
