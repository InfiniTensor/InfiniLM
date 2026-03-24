#pragma once

#include "../../config/infinilm_config.hpp"
#include "../../config/model_config.hpp"
#include "../../engine/parallel_state.hpp"
#include "../../layers/common_modules.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include "../../engine/distributed/distributed.hpp"

#include <cmath>
#include <memory>

namespace infinilm::models::qwen3_next {

class Qwen3NextAttention : public infinicore::nn::Module {
public:
    Qwen3NextAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       size_t layer_idx,
                       const infinicore::Device &device) {
        layer_idx_ = layer_idx;
        const auto &dtype{model_config->get_dtype()};

        num_attention_heads_ = model_config->get<size_t>("num_attention_heads");
        num_key_value_heads_ = model_config->get<size_t>("num_key_value_heads");
        hidden_size_ = model_config->get<size_t>("hidden_size");
        head_dim_ = model_config->get<size_t>("head_dim");

        float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));

        bool use_bias = model_config->get_or<bool>("attention_bias", true);
        bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");
        bool attn_output_gate = model_config->get_or<bool>("attn_output_gate", true);

        auto quant_scheme = model_config->get_quant_scheme();
        auto quantization_method = model_config->get_quantization_method();

        const engine::distributed::RankInfo &rank_info = infinilm::engine::get_tensor_model_parallel_rank_info();
        int tp_rank = infinilm::engine::get_tensor_model_parallel_rank();
        int tp_size = infinilm::engine::get_tensor_model_parallel_world_size();

        size_t total_num_heads = num_attention_heads_;

        switch (quant_scheme) {
        case infinicore::quantization::QuantScheme::NONE: {
            INFINILM_QKV_LINEAR_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, total_num_heads * (1 + attn_output_gate), num_key_value_heads_, quantization_method, use_bias, dtype, device, rank_info);
            INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method,
                                      use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
            break;
        }
        case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
            INFINILM_QKV_LINEAR_W8A8_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, total_num_heads * (1 + attn_output_gate), num_key_value_heads_, quantization_method, use_bias, dtype, device, rank_info);
            INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method,
                                      use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
            break;
        }
        case infinicore::quantization::QuantScheme::AWQ_W4A16: {
            INFINILM_QKV_LINEAR_W4A16AWQ_INIT(qkv_proj, "q_proj", "k_proj", "v_proj", hidden_size_, head_dim_, total_num_heads * (1 + attn_output_gate), num_key_value_heads_, quantization_method, use_bias, dtype, device, rank_info);
            INFINICORE_NN_MODULE_INIT(o_proj, total_num_heads * head_dim_, hidden_size_, quantization_method,
                                      use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);
            break;
        }
        default: {
            throw std::runtime_error("infinilm::models::qwen3_next::Qwen3NextAttention: unsupported quantization scheme");

            break;
        }
        }

        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);

        if ((num_key_value_heads_ < tp_size) || (0 != (num_key_value_heads_ % tp_size))) {
            throw std::runtime_error("infinilm::models::qwen3_next::Qwen3NextAttention: num_key_value_heads must be divisible by tp_size");
        }

        attention_backend_ = infinilm::config::get_current_infinilm_config().attention_backend;
        size_t num_attention_heads_rank = num_attention_heads_ / tp_size;
        size_t num_key_value_heads_rank = num_key_value_heads_ / tp_size;
        attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
            num_attention_heads_rank, head_dim_, scaling, num_key_value_heads_rank, layer_idx_, attention_backend_);

        num_attention_heads_ = num_attention_heads_rank;
        num_key_value_heads_ = num_key_value_heads_rank;
    }

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const {
        spdlog::error("infinilm::models::qwen3_next::Qwen3NextAttention: forward not implemented");
        return hidden_states;
    }

public:
    size_t layer_idx() const { return layer_idx_; }
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }
    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) { rotary_emb_ = rotary_emb; }

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::QKVParallelLinear, qkv_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);

    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

    std::shared_ptr<infinilm::layers::attention::AttentionLayer> attn_;
    ::infinilm::backends::AttentionBackend attention_backend_;
    size_t layer_idx_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t hidden_size_;
    size_t head_dim_;
};

} // namespace infinilm::models::qwen3_next
