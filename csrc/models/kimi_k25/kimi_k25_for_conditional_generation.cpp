#include "../../layers/rotary_embedding/rope.hpp"
#include "kimi_k25_for_conditional_generation.hpp"
#include <cmath>
#include <exception>
#include "runtime/runtime.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infinilm::models::kimi_k25 {

KimiK25ForConditionalGeneration::KimiK25ForConditionalGeneration(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    try {
    model_config_ = model_config;
    const auto &dtype = model_config->get_dtype();
    const auto &vision_config = model_config->get_config_json().at("vision_config");
        vision_tower_ = std::make_shared<std::remove_reference_t<decltype(*vision_tower_)>>(
            runtime, vision_config, dtype);
        this->bind_child("vision_tower", vision_tower_);
        mm_projector_ = std::make_shared<std::remove_reference_t<decltype(*mm_projector_)>>(
            runtime, vision_config, dtype);
        this->bind_child("mm_projector", mm_projector_);
        language_model_ = std::make_shared<std::remove_reference_t<decltype(*language_model_)>>(
            runtime, model_config, rank_info, attention_backend, ep_config);
        this->bind_child("language_model", language_model_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

void KimiK25ForConditionalGeneration::replace_image_embeddings(
    infinilm::runtime::Tensor &inputs_embeds,
    const Input &input,
    global_state::ForwardContext &forward_context) const {
    if (!input.pixel_values.has_value() || input.pixel_values->empty()) {
        return;
    }
    if (!input.image_grid_thw.has_value() || !input.image_bound.has_value()
        || !input.input_offsets.has_value()) {
        throw std::runtime_error(
            "KimiK25ForConditionalGeneration: image_grid_thw, image_bound and input_offsets are required");
    }
    const auto &request_ids = forward_context.mm_metadata.image_req_ids;
    if (!request_ids.has_value() || request_ids->size() != input.pixel_values->size()) {
        throw std::runtime_error("KimiK25ForConditionalGeneration: image_req_ids do not match pixel_values");
    }

    auto &runtime = forward_context.require_runtime();
    const auto cpu_device = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    auto offsets_cpu = input.input_offsets.value();
    if (offsets_cpu->device() != cpu_device) {
        offsets_cpu = offsets_cpu->to(runtime, cpu_device);
    }
    if (offsets_cpu->dtype() != infinilm::runtime::DataType::kInt32) {
        throw std::runtime_error("KimiK25ForConditionalGeneration: input_offsets must be int32");
    }
    const auto *offsets = reinterpret_cast<const int32_t *>(offsets_cpu->data());

    for (size_t image_idx = 0; image_idx < input.pixel_values->size(); ++image_idx) {
        auto bound_cpu = input.image_bound->at(image_idx);
        if (bound_cpu->device() != cpu_device) {
            bound_cpu = bound_cpu->to(runtime, cpu_device);
        }
        if (bound_cpu->dtype() != infinilm::runtime::DataType::kInt64 || bound_cpu->numel() != 2) {
            throw std::runtime_error("KimiK25ForConditionalGeneration: image_bound must be int64 [2]");
        }
        const auto *bound = reinterpret_cast<const int64_t *>(bound_cpu->data());
        const size_t request_id = request_ids->at(image_idx);
        const size_t request_start = static_cast<size_t>(offsets[request_id]);
        const size_t image_start = static_cast<size_t>(bound[0]);
        const size_t image_length = static_cast<size_t>(bound[1] - bound[0]);

        auto vision_features = vision_tower_->forward(runtime,
            input.pixel_values->at(image_idx), input.image_grid_thw->at(image_idx));
        auto image_embeds = mm_projector_->forward(runtime, vision_features);
        if (image_embeds->size(0) != image_length) {
            throw std::runtime_error("KimiK25ForConditionalGeneration: image token span does not match projector output");
        }
        inputs_embeds->narrow({{1, request_start + image_start, image_length}})
            ->copy_from(runtime, image_embeds->unsqueeze(0));
    }
}

InfinilmModel::Output KimiK25ForConditionalGeneration::forward(
    const Input &input, global_state::ForwardContext &forward_context) const {
    if (forward_context.runtime == nullptr) {
        throw std::invalid_argument("KimiK25ForConditionalGeneration: Runtime is required");
    }
    forward_context.runtime->assertOwnerThread();
    if (!input.pixel_values.has_value() || input.pixel_values->empty()) {
        return language_model_->forward(input, forward_context);
    }
    auto inputs_embeds = language_model_->model().embed_tokens(input.input_ids.value(), forward_context);
    replace_image_embeddings(inputs_embeds, input, forward_context);
    auto hidden_states = language_model_->model().forward_embeds(
        inputs_embeds, input.position_ids.value(), forward_context);
    return {language_model_->logits_from_hidden(forward_context.require_runtime(), hidden_states)};
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
    if (!config_json.at("head_dim").get<size_t>())
        throw std::invalid_argument("kimi_k25 requires nonzero attention head dimension");
    config_json["partial_rotary_factor"] = static_cast<double>(config_json.at("qk_rope_head_dim").get<size_t>())
                                         / static_cast<double>(config_json.at("head_dim").get<size_t>());
    config_json["num_experts"] = config_json.at("n_routed_experts");
    config_json["mlp_bias"] = false;
    config_json["attention_output_bias"] = config_json.value("attention_bias", false);
    config_json["e_score_correction_bias"] = true;
    config_json["apply_routed_scaling_factor_on_output"] = false;

    // Only metadata consumed by the actual construction branch is required.
    if (model_config->get<int64_t>("hidden_size") <= 0
        || model_config->get<int64_t>("vocab_size") <= 0
        || model_config->get<int64_t>("num_hidden_layers") < 0)
        throw std::invalid_argument("Invalid kimi_k25 construction dimensions");
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    const size_t first_dense = model_config->get_or<size_t>("first_k_dense_replace", 0);
    const size_t frequency = model_config->get_or<size_t>("moe_layer_freq", 1);
    bool needs_dense = false;
    for (size_t i = 0; i < num_layers; ++i) {
        const bool moe = model_config->get_or<size_t>("n_routed_experts", 0) > 0
                      && i >= first_dense && (frequency == 0 || i % frequency == 0);
        needs_dense = needs_dense || !moe;
    }
    if (needs_dense && model_config->get<int64_t>("intermediate_size") <= 0)
        throw std::invalid_argument("kimi_k25 dense layers require positive intermediate_size");

    model_config->set_rope_algo(infinilm::layers::RoPE::Algo::GPT_J);
    return model_config;
}

} // namespace infinilm::models::kimi_k25

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    kimi_k25,
    infinilm::models::kimi_k25::KimiK25ForConditionalGeneration,
    infinilm::models::kimi_k25::create_kimi_k25_model_config);
} // namespace
