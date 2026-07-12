#include "ernie4_5_moe_vl_for_causal_lm.hpp"

#include "../models_registry.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/cat.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

namespace infinilm::models::ernie4_5_moe_vl {

Ernie4_5Model::Ernie4_5Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);

    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<Ernie4_5DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }

    INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(resampler_model, model_config, device);
}

infinicore::Tensor Ernie4_5Model::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = embed_tokens_->forward(input_ids);
    return forward_embeds(hidden_states, positions, input.token_type_ids.value_or(infinicore::Tensor()));
}

infinicore::Tensor Ernie4_5Model::forward_embeds(const infinicore::Tensor &inputs_embeds,
                                                 const infinicore::Tensor &position_ids,
                                                 const infinicore::Tensor &token_type_ids) const {
    auto hidden_states = inputs_embeds;
    infinicore::Tensor residual;
    for (const auto &layer : layers_) {
        layer->forward(position_ids, hidden_states, residual, token_type_ids);
    }
    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

infinicore::Tensor Ernie4_5Model::embed_tokens(const infinicore::Tensor &input_ids) const {
    return embed_tokens_->forward(input_ids);
}

infinicore::Tensor Ernie4_5Model::resample_vision(const infinicore::Tensor &vision_features,
                                                  const infinicore::Tensor &grid_thw) const {
    return resampler_model_->forward(vision_features, grid_thw);
}

Ernie4_5MoeVLForCausalLM::Ernie4_5MoeVLForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                   const infinicore::Device &device) {
    model_config_ = model_config;

    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype{model_config->get_dtype()};
    im_patch_id_ = model_config->get_or<size_t>("im_patch_id", 100295);

    INFINICORE_NN_MODULE_INIT(vision_model, model_config, device);
    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

InfinilmModel::Output Ernie4_5MoeVLForCausalLM::forward(const Input &input) const {
    if (input.pixel_values.has_value() && !input.pixel_values->empty()) {
        auto inputs_embeds = build_multimodal_embeds_(input);
        auto hidden_states = model_->forward_embeds(
            inputs_embeds,
            input.position_ids.value(),
            input.token_type_ids.value_or(infinicore::Tensor()));
        auto logits = lm_head_->forward(hidden_states);
        return {logits};
    }
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

infinicore::Tensor Ernie4_5MoeVLForCausalLM::logits_from_hidden(const infinicore::Tensor &hidden_states) const {
    return lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states));
}

infinicore::Tensor Ernie4_5MoeVLForCausalLM::concat_optional_tensors_(
    const std::optional<std::vector<infinicore::Tensor>> &tensors,
    int dim) const {
    if (!tensors.has_value() || tensors->empty()) {
        return infinicore::Tensor();
    }
    if (tensors->size() == 1) {
        return tensors->front();
    }
    return infinicore::op::cat(*tensors, dim);
}

infinicore::Tensor Ernie4_5MoeVLForCausalLM::replace_image_embeds_(const infinicore::Tensor &input_ids,
                                                                   const infinicore::Tensor &inputs_embeds,
                                                                   const infinicore::Tensor &image_features) const {
    ASSERT(input_ids->ndim() == 2);
    ASSERT(inputs_embeds->ndim() == 3);
    ASSERT(image_features->ndim() == 2);
    ASSERT_EQ(input_ids->shape()[0], inputs_embeds->shape()[0]);
    ASSERT_EQ(input_ids->shape()[1], inputs_embeds->shape()[1]);
    ASSERT_EQ(inputs_embeds->shape()[2], image_features->shape()[1]);

    auto ids_cpu = input_ids->to(infinicore::Device::cpu());
    auto embeds_cpu = inputs_embeds->to(infinicore::Device::cpu());
    auto features_cpu = image_features->to(infinicore::Device::cpu());
    auto out_cpu = infinicore::Tensor::empty(inputs_embeds->shape(), inputs_embeds->dtype(), infinicore::Device::cpu());
    out_cpu->copy_from(embeds_cpu);

    const auto *ids = reinterpret_cast<const int64_t *>(ids_cpu->data());
    const auto *features = reinterpret_cast<const std::byte *>(features_cpu->data());
    auto *out = reinterpret_cast<std::byte *>(out_cpu->data());
    const size_t batch = input_ids->shape()[0];
    const size_t seq_len = input_ids->shape()[1];
    const size_t hidden = inputs_embeds->shape()[2];
    const size_t elem_size = inputs_embeds->element_size();
    const size_t row_bytes = hidden * elem_size;

    size_t image_row = 0;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            if (ids[b * seq_len + s] != static_cast<int64_t>(im_patch_id_)) {
                continue;
            }
            if (image_row >= image_features->shape()[0]) {
                throw std::runtime_error("Ernie4_5MoeVLForCausalLM: fewer image features than image patch tokens");
            }
            const size_t dst_row = (b * seq_len + s) * hidden;
            std::memcpy(out + dst_row * elem_size, features + image_row * row_bytes, row_bytes);
            ++image_row;
        }
    }
    if (image_row != image_features->shape()[0]) {
        throw std::runtime_error("Ernie4_5MoeVLForCausalLM: image feature count does not match image patch tokens");
    }

    return out_cpu->to(inputs_embeds->device());
}

infinicore::Tensor Ernie4_5MoeVLForCausalLM::build_multimodal_embeds_(const Input &input) const {
    if (!input.input_ids.has_value() || !input.position_ids.has_value()) {
        throw std::runtime_error("Ernie4_5MoeVLForCausalLM: input_ids and position_ids are required");
    }
    auto images = concat_optional_tensors_(input.pixel_values, 0);
    auto grid_thw = concat_optional_tensors_(input.grid_thw, 0);
    if (!images || !grid_thw) {
        throw std::runtime_error("Ernie4_5MoeVLForCausalLM: images and grid_thw are required for vision input");
    }

    auto input_ids = input.input_ids.value();
    auto inputs_embeds = model_->embed_tokens(input_ids);
    auto image_features = vision_model_->forward(images, grid_thw);
    image_features = model_->resample_vision(image_features, grid_thw);
    return replace_image_embeds_(input_ids, inputs_embeds, image_features);
}

std::shared_ptr<infinilm::config::ModelConfig>
create_ernie4_5_moe_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("ernie4_5_moe_vl" != model_type) {
        throw std::runtime_error("create_ernie4_5_moe_vl_model_config: model_type is not ernie4_5_moe_vl");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = model_config->get<size_t>("hidden_size")
                                / model_config->get<size_t>("num_attention_heads");
    }
    if (!config_json.contains("num_key_value_heads") || config_json["num_key_value_heads"].is_null()) {
        config_json["num_key_value_heads"] = model_config->get<size_t>("num_attention_heads");
    }
    if (!config_json.contains("rope_theta")) {
        config_json["rope_theta"] = 10000.0;
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = config_json.value("use_bias", false);
    }
    if (!config_json.contains("attention_output_bias")) {
        config_json["attention_output_bias"] = config_json.value("use_bias", false);
    }
    if (!config_json.contains("mlp_bias")) {
        config_json["mlp_bias"] = config_json.value("use_bias", false);
    }
    if (!config_json.contains("moe_k")) {
        config_json["moe_k"] = 6;
    }
    if (!config_json.contains("moe_num_shared_experts")) {
        config_json["moe_num_shared_experts"] = 0;
    }
    if (!config_json.contains("moe_norm_gate_logits")) {
        config_json["moe_norm_gate_logits"] = true;
    }
    if (!config_json.contains("moe_use_aux_free")) {
        config_json["moe_use_aux_free"] = false;
    }

    // ERNIE applies RoPE on adjacent even/odd pairs, matching InfiniCore's GPT-J layout.
    model_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);

    return model_config;
}

} // namespace infinilm::models::ernie4_5_moe_vl

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    ernie4_5_moe_vl,
    infinilm::models::ernie4_5_moe_vl::Ernie4_5MoeVLForCausalLM,
    infinilm::models::ernie4_5_moe_vl::create_ernie4_5_moe_vl_model_config);
} // namespace
