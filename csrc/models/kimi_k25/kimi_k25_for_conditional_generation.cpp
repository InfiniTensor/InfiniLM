#include "kimi_k25_for_conditional_generation.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infinilm::models::kimi_k25 {

KimiK25ForConditionalGeneration::KimiK25ForConditionalGeneration(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    const auto &dtype = model_config->get_dtype();
    const auto &vision_config = model_config->get_config_json().at("vision_config");
    INFINICORE_NN_MODULE_INIT(vision_tower, vision_config, dtype, device);
    INFINICORE_NN_MODULE_INIT(mm_projector, vision_config, dtype, device);
    INFINICORE_NN_MODULE_INIT(language_model, model_config, device);
}

void KimiK25ForConditionalGeneration::replace_image_embeddings(
    infinicore::Tensor &inputs_embeds,
    const Input &input) const {
    if (!input.pixel_values.has_value() || input.pixel_values->empty()) {
        return;
    }
    if (!input.image_grid_thw.has_value() || !input.image_bound.has_value()
        || !input.input_offsets.has_value()) {
        throw std::runtime_error(
            "KimiK25ForConditionalGeneration: image_grid_thw, image_bound and input_offsets are required");
    }
    const auto &request_ids = global_state::get_forward_context().mm_metadata.image_req_ids;
    if (!request_ids.has_value() || request_ids->size() != input.pixel_values->size()) {
        throw std::runtime_error("KimiK25ForConditionalGeneration: image_req_ids do not match pixel_values");
    }

    auto offsets_cpu = input.input_offsets.value()->to(infinicore::Device::cpu());
    if (offsets_cpu->dtype() != infinicore::DataType::I32) {
        throw std::runtime_error("KimiK25ForConditionalGeneration: input_offsets must be int32");
    }
    const auto *offsets = reinterpret_cast<const int32_t *>(offsets_cpu->data());

    for (size_t image_idx = 0; image_idx < input.pixel_values->size(); ++image_idx) {
        auto bound_cpu = input.image_bound->at(image_idx)->to(infinicore::Device::cpu());
        if (bound_cpu->dtype() != infinicore::DataType::I64 || bound_cpu->numel() != 2) {
            throw std::runtime_error("KimiK25ForConditionalGeneration: image_bound must be int64 [2]");
        }
        const auto *bound = reinterpret_cast<const int64_t *>(bound_cpu->data());
        const size_t request_id = request_ids->at(image_idx);
        const size_t request_start = static_cast<size_t>(offsets[request_id]);
        const size_t image_start = static_cast<size_t>(bound[0]);
        const size_t image_length = static_cast<size_t>(bound[1] - bound[0]);

        auto vision_features = vision_tower_->forward(
            input.pixel_values->at(image_idx), input.image_grid_thw->at(image_idx));
        auto image_embeds = mm_projector_->forward(vision_features);
        if (image_embeds->size(0) != image_length) {
            throw std::runtime_error("KimiK25ForConditionalGeneration: image token span does not match projector output");
        }
        inputs_embeds->narrow({{1, request_start + image_start, image_length}})
            ->copy_from(image_embeds->unsqueeze(0));
    }
}

InfinilmModel::Output KimiK25ForConditionalGeneration::forward(const Input &input) const {
    if (!input.pixel_values.has_value() || input.pixel_values->empty()) {
        return language_model_->forward(input);
    }
    auto inputs_embeds = language_model_->model().embed_tokens(input.input_ids.value());
    replace_image_embeddings(inputs_embeds, input);
    auto hidden_states = language_model_->model().forward_embeds(
        inputs_embeds, input.position_ids.value());
    return {language_model_->logits_from_hidden(hidden_states)};
}

void KimiK25ForConditionalGeneration::reset_cache(const cache::CacheConfig *cache_config) {
    InfinilmModel::reset_cache(cache_config);
}

std::shared_ptr<infinilm::config::ModelConfig>
create_kimi_k25_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if (model_type != "kimi_k25") {
        throw std::runtime_error("create_kimi_k25_model_config: model_type is not kimi_k25");
    }

    auto &config_json = model_config->get_config_json();
    const auto text_config = config_json.at("text_config");
    for (auto it = text_config.begin(); it != text_config.end(); ++it) {
        if (!config_json.contains(it.key()) || config_json.at(it.key()).is_null()) {
            config_json[it.key()] = it.value();
        }
    }
    config_json["head_dim"] = config_json.at("qk_nope_head_dim").get<size_t>()
                            + config_json.at("qk_rope_head_dim").get<size_t>();
    config_json["partial_rotary_factor"] = static_cast<double>(config_json.at("qk_rope_head_dim").get<size_t>())
                                         / static_cast<double>(config_json.at("head_dim").get<size_t>());
    config_json["num_experts"] = config_json.at("n_routed_experts");
    config_json["mlp_bias"] = false;
    config_json["attention_output_bias"] = config_json.value("attention_bias", false);
    config_json["e_score_correction_bias"] = true;
    config_json["apply_routed_scaling_factor_on_output"] = false;
    // TODO(kimi_k25): Restore the grouped noaux_tc router after the graph-backed
    // MoeFusedGate path is safe for multi-rank model execution.
    config_json["moe_router_backend"] = "sigmoid";

    model_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);
    return model_config;
}

} // namespace infinilm::models::kimi_k25

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    kimi_k25,
    infinilm::models::kimi_k25::KimiK25ForConditionalGeneration,
    infinilm::models::kimi_k25::create_kimi_k25_model_config);
} // namespace
