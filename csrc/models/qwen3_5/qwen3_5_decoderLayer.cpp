#include "qwen3_5_decoderLayer.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/add_rms_norm.hpp"
#include "infinicore/ops/cast.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5 {
namespace {

void dump_prefill_tensor(const infinicore::Tensor &tensor,
                         const std::string &filename) {
    const char *dump_dir = std::getenv("INFINILM_LAYER_DUMP_DIR");
    if (dump_dir == nullptr || dump_dir[0] == '\0' || !tensor) {
        return;
    }
    const char *dump_numel = std::getenv("INFINILM_LAYER_DUMP_NUMEL");
    if (dump_numel == nullptr || dump_numel[0] == '\0'
        || tensor->numel() != std::strtoull(dump_numel, nullptr, 10)) {
        return;
    }
    tensor->debug(std::string(dump_dir) + "/" + filename);
}

bool should_dump_layer(size_t layer_idx) {
    const char *first_n = std::getenv("INFINILM_LAYER_DUMP_FIRST_N");
    if (first_n != nullptr && first_n[0] != '\0'
        && layer_idx < std::strtoull(first_n, nullptr, 10)) {
        return true;
    }
    return (layer_idx + 1) % 8 == 0;
}

bool should_dump_operators(size_t layer_idx) {
    const char *target = std::getenv("INFINILM_OPERATOR_DUMP_LAYER");
    return target != nullptr && target[0] != '\0'
        && layer_idx == std::strtoull(target, nullptr, 10);
}

} // namespace

Qwen35DecoderLayer::Qwen35DecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       size_t layer_idx,
                                       const infinicore::Device &device)
    : layer_idx_(layer_idx) {

    const auto &dtype{model_config->get_dtype()};
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(input_layernorm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, hidden_size, rms_norm_eps, dtype, device);
    // Checkpoint path prefix for quantization schemes that resolve types by
    // tensor name.
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device, "layers." + std::to_string(layer_idx) + ".mlp");

    const std::vector<std::string> layer_types = model_config->get<std::vector<std::string>>("layer_types");
    layer_type_ = layer_types[layer_idx];
    if ("linear_attention" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(linear_attn, model_config, layer_idx, device);
    } else if ("full_attention" == layer_type_) {
        INFINICORE_NN_MODULE_INIT(self_attn, model_config, layer_idx, device);
    } else {
        throw std::runtime_error("infinilm::models::qwen3_5::Qwen35DecoderLayer: unsupported layer_type '" + layer_type_ + "' for layer " + std::to_string(layer_idx));
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> Qwen35DecoderLayer::forward(const infinicore::Tensor &positions,
                                                                               infinicore::Tensor &hidden_states,
                                                                               infinicore::Tensor &residual) {
    if (layer_idx_ == 0) {
        dump_prefill_tensor(hidden_states, "infini_embed.bin");
    }
    if (residual
        && hidden_states->dtype() == infinicore::DataType::F32
        && residual->dtype() == infinicore::DataType::BF16) {
        auto y = infinicore::Tensor::empty(
            hidden_states->shape(), infinicore::DataType::BF16, hidden_states->device());
        auto residual_out = infinicore::Tensor::empty(
            residual->shape(), infinicore::DataType::BF16, residual->device());
        infinicore::op::add_rms_norm_(
            y, residual_out, hidden_states, residual,
            input_layernorm_->weight(),
            static_cast<float>(input_layernorm_->eps()));
        hidden_states = y;
        residual = residual_out;
    } else {
        input_layernorm_->forward_inplace(hidden_states, residual);
    }
    if ("linear_attention" == layer_type_) {
        hidden_states = linear_attn_->forward(hidden_states);
    } else if ("full_attention" == layer_type_) {
        hidden_states = self_attn_->forward(positions, hidden_states);
    }

    const char *fp32_fused_env = std::getenv("INFINILM_POST_NORM_FP32_FUSED");
    const bool fp32_fused = fp32_fused_env != nullptr && fp32_fused_env[0] != '\0'
                         && std::string(fp32_fused_env) != "0";
    const bool mixed_gguf_f32 = residual
                             && hidden_states->dtype() == infinicore::DataType::F32
                             && residual->dtype() == infinicore::DataType::BF16;
    if (mixed_gguf_f32) {
        auto y = infinicore::Tensor::empty(
            hidden_states->shape(), infinicore::DataType::BF16, hidden_states->device());
        auto residual_out = infinicore::Tensor::empty(
            residual->shape(), infinicore::DataType::BF16, residual->device());
        infinicore::op::add_rms_norm_(
            y, residual_out, hidden_states, residual,
            post_attention_layernorm_->weight(),
            static_cast<float>(post_attention_layernorm_->eps()));
        hidden_states = y;
        residual = residual_out;
    } else if (fp32_fused) {
        auto a32 = infinicore::Tensor::empty(hidden_states->shape(), infinicore::DataType::F32, hidden_states->device());
        auto b32 = infinicore::Tensor::empty(residual->shape(), infinicore::DataType::F32, residual->device());
        infinicore::op::cast_(a32, hidden_states);
        infinicore::op::cast_(b32, residual);
        auto y32 = infinicore::Tensor::empty(hidden_states->shape(), infinicore::DataType::F32, hidden_states->device());
        auto r32 = infinicore::Tensor::empty(residual->shape(), infinicore::DataType::F32, residual->device());
        infinicore::op::add_rms_norm_(y32, r32, a32, b32,
                                      post_attention_layernorm_->weight(),
                                      static_cast<float>(post_attention_layernorm_->eps()));
        hidden_states = y32;
        residual = r32;
    } else {
        post_attention_layernorm_->forward_inplace(hidden_states, residual);
    }
    if (should_dump_operators(layer_idx_)) {
        dump_prefill_tensor(residual,
                            "infini_attn_residual_" + std::to_string(layer_idx_) + ".bin");
        dump_prefill_tensor(hidden_states,
                            "infini_attn_post_norm_" + std::to_string(layer_idx_) + ".bin");
    }
    const char *fp32_mlp_env = std::getenv("INFINILM_POST_NORM_FP32_MLP");
    const bool fp32_mlp = fp32_mlp_env != nullptr && fp32_mlp_env[0] != '\0'
                       && std::string(fp32_mlp_env) != "0";
    if (fp32_mlp && !fp32_fused) {
        auto fp32_hidden = infinicore::Tensor::empty(
            hidden_states->shape(), infinicore::DataType::F32, hidden_states->device());
        infinicore::op::cast_(fp32_hidden, hidden_states);
        hidden_states = fp32_hidden;
    }
    hidden_states = mlp_->forward(hidden_states);
    if (should_dump_operators(layer_idx_)) {
        dump_prefill_tensor(hidden_states,
                            "infini_ffn_out_" + std::to_string(layer_idx_) + ".bin");
    }
    if (fp32_mlp && !fp32_fused) {
        auto bf16_hidden = infinicore::Tensor::empty(
            hidden_states->shape(), infinicore::DataType::BF16, hidden_states->device());
        infinicore::op::cast_(bf16_hidden, hidden_states);
        hidden_states = bf16_hidden;
    }
    if (should_dump_layer(layer_idx_)) {
        auto materialized = residual ? infinicore::op::add(residual, hidden_states)
                                     : hidden_states;
        dump_prefill_tensor(materialized,
                            "infini_layer_" + std::to_string(layer_idx_) + "_post_ffn.bin");
    }
    if (fp32_fused) {
        auto bf16_hidden = infinicore::Tensor::empty(
            hidden_states->shape(), infinicore::DataType::BF16, hidden_states->device());
        infinicore::op::cast_(bf16_hidden, hidden_states);
        hidden_states = bf16_hidden;
        auto bf16_residual = infinicore::Tensor::empty(
            residual->shape(), infinicore::DataType::BF16, residual->device());
        infinicore::op::cast_(bf16_residual, residual);
        residual = bf16_residual;
    }
    return std::make_tuple(hidden_states, residual);
}

infinicore::Tensor Qwen35DecoderLayer::forward(const infinicore::Tensor &positions,
                                               infinicore::Tensor &hidden_states) {
    auto residual = hidden_states;
    hidden_states = input_layernorm_->forward(hidden_states);
    if ("linear_attention" == layer_type_) {
        hidden_states = linear_attn_->forward(hidden_states);
    } else if ("full_attention" == layer_type_) {
        hidden_states = self_attn_->forward(positions, hidden_states);
    }
    hidden_states = infinicore::op::add(residual, hidden_states);

    residual = hidden_states;
    hidden_states = post_attention_layernorm_->forward(hidden_states);
    hidden_states = mlp_->forward(hidden_states);
    hidden_states = infinicore::op::add(residual, hidden_states);
    return hidden_states;
}

} // namespace infinilm::models::qwen3_5
