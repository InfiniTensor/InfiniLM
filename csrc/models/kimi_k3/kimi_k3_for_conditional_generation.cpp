#include "kimi_k3_for_conditional_generation.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "kimi_k3_allocate_cache.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::kimi_k3 {

KimiK3ForConditionalGeneration::KimiK3ForConditionalGeneration(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    const auto &rank_info = global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.pp_stage == 0) {
        const auto &vision_config = model_config->get_config_json().at("vision_config");
        INFINICORE_NN_MODULE_INIT(vision_tower, vision_config, model_config->get_dtype(), device);
        INFINICORE_NN_MODULE_INIT(mm_projector, vision_config, model_config->get_dtype(), device);
    }
    INFINICORE_NN_MODULE_INIT(language_model, model_config, device);
}

void KimiK3ForConditionalGeneration::replace_image_embeddings(
    infinicore::Tensor &inputs_embeds,
    const Input &input) const {
    if (!input.pixel_values.has_value() || input.pixel_values->empty()) {
        return;
    }
    if (!vision_tower_ || !mm_projector_) {
        throw std::runtime_error("KimiK3: vision input reached a non-first PP stage");
    }
    if (!input.image_grid_thw.has_value() || !input.image_bound.has_value()
        || !input.input_offsets.has_value()) {
        throw std::runtime_error("KimiK3: image_grid_thw, image_bound and input_offsets are required");
    }
    const auto &request_ids = global_state::get_forward_context().mm_metadata.image_req_ids;
    if (!request_ids.has_value() || request_ids->size() != input.pixel_values->size()) {
        throw std::runtime_error("KimiK3: image request metadata does not match pixel_values");
    }
    auto offsets_cpu = input.input_offsets.value()->to(infinicore::Device::cpu());
    const auto *offsets = reinterpret_cast<const int32_t *>(offsets_cpu->data());
    for (size_t image_idx = 0; image_idx < input.pixel_values->size(); ++image_idx) {
        auto bound_cpu = input.image_bound->at(image_idx)->to(infinicore::Device::cpu());
        const auto *bound = reinterpret_cast<const int64_t *>(bound_cpu->data());
        const size_t request_start = static_cast<size_t>(offsets[request_ids->at(image_idx)]);
        const size_t image_start = static_cast<size_t>(bound[0]);
        const size_t image_length = static_cast<size_t>(bound[1] - bound[0]);
        auto features = vision_tower_->forward(
            input.pixel_values->at(image_idx), input.image_grid_thw->at(image_idx));
        auto image_embeds = mm_projector_->forward(features);
        if (image_embeds->size(0) != image_length) {
            throw std::runtime_error("KimiK3: projected image length does not match image_bound");
        }
        inputs_embeds->narrow({{1, request_start + image_start, image_length}})
            ->copy_from(image_embeds->unsqueeze(0));
    }
}

InfinilmModel::Output KimiK3ForConditionalGeneration::forward(const Input &input) const {
    if (!input.pixel_values.has_value() || input.pixel_values->empty()
        || !language_model_->model().is_first_pp_stage()) {
        return language_model_->forward(input);
    }
    auto inputs_embeds = language_model_->model().embed_tokens(input.input_ids.value());
    replace_image_embeddings(inputs_embeds, input);
    auto hidden_states = language_model_->model().forward_embeds(inputs_embeds);
    if (!language_model_->model().is_last_pp_stage()) {
        return {infinicore::Tensor(), hidden_states};
    }
    return {language_model_->logits_from_hidden(hidden_states), hidden_states};
}

void KimiK3ForConditionalGeneration::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        return;
    }
    auto allocated = kimi_k3_allocate_cache_tensors(
        cache_config,
        model_config_,
        global_state::get_infinilm_config().attention_backend);
    auto &context = global_state::get_forward_context();
    context.kv_cache_vec = std::move(allocated.kv_cache_tensors);
    context.conv_state_vec = std::move(allocated.conv_state_tensors);
    context.ssm_state_vec = std::move(allocated.ssm_state_tensors);
}

std::shared_ptr<infinilm::config::ModelConfig>
create_kimi_k3_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (model_config->get<std::string>("model_type") != "kimi_k3") {
        throw std::runtime_error("create_kimi_k3_model_config: model_type is not kimi_k3");
    }
    auto &config = model_config->get_config_json();
    const auto text_config = config.at("text_config");
    for (auto it = text_config.begin(); it != text_config.end(); ++it) {
        if (it.key() != "model_type") {
            config[it.key()] = it.value();
        }
    }
    // K3's compressed-tensors ignore list leaves attention, shared MLPs,
    // vision, and the LM head in BF16. Routed experts load MXFP4 explicitly.
    config["quantization_config"] = nullptr;
    config["head_dim"] = config.at("qk_nope_head_dim").get<size_t>()
                       + config.at("qk_rope_head_dim").get<size_t>();
    config["num_experts_per_tok"] = config.at("num_experts_per_token");
    config["norm_topk_prob"] = config.at("moe_renormalize");
    config["moe_router_backend"] = "sigmoid";
    config["e_score_correction_bias"] = true;
    config["apply_routed_scaling_factor_on_output"] = false;
    config["num_fused_shared_experts"] = 0;
    config["mlp_bias"] = false;

    const size_t num_layers = config.at("num_hidden_layers").get<size_t>();
    std::vector<std::string> layer_types(num_layers, "full_attention");
    for (const size_t one_based_layer_idx :
         config.at("linear_attn_config").at("kda_layers").get<std::vector<size_t>>()) {
        layer_types.at(one_based_layer_idx - 1) = "linear_attention";
    }
    config["layer_types"] = std::move(layer_types);
    return model_config;
}

} // namespace infinilm::models::kimi_k3

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    kimi_k3,
    infinilm::models::kimi_k3::KimiK3ForConditionalGeneration,
    infinilm::models::kimi_k3::create_kimi_k3_model_config);
} // namespace
