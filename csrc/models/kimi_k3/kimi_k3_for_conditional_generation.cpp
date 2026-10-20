#include "kimi_k3_for_conditional_generation.hpp"
#include <cmath>
#include <exception>

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "kimi_k3_allocate_cache.hpp"
#include "runtime/runtime.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::kimi_k3 {

KimiK3ForConditionalGeneration::KimiK3ForConditionalGeneration(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    try {
        model_config_ = model_config;
        owner_tp_size_ = rank_info.tp_size;
        owner_pp_size_ = rank_info.pp_size;
        owner_pp_stage_ = rank_info.pp_stage;
        if (rank_info.pp_stage == 0) {
            const auto &vision_config = model_config->get_config_json().at("vision_config");
            vision_tower_ = std::make_shared<std::remove_reference_t<decltype(*vision_tower_)>>(
                runtime, vision_config, model_config->get_dtype());
            this->bind_child("vision_tower", vision_tower_);
            mm_projector_ = std::make_shared<std::remove_reference_t<decltype(*mm_projector_)>>(
                runtime, vision_config, model_config->get_dtype());
            this->bind_child("mm_projector", mm_projector_);
        }
        language_model_ = std::make_shared<std::remove_reference_t<decltype(*language_model_)>>(
            runtime, model_config, rank_info, attention_backend, ep_config);
        this->bind_child("language_model", language_model_);
    } catch (...) {
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        throw;
    }
}

void KimiK3ForConditionalGeneration::replace_image_embeddings(
    infinilm::runtime::Tensor &inputs_embeds,
    const Input &input,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
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
    const auto &request_ids = forward_context.mm_metadata.image_req_ids;
    if (!request_ids.has_value() || request_ids->size() != input.pixel_values->size()) {
        throw std::runtime_error("KimiK3: image request metadata does not match pixel_values");
    }
    auto offsets_cpu = input.input_offsets.value()->device().type() == infinilm::runtime::Device::Type::kCpu
                         ? input.input_offsets.value()
                         : input.input_offsets.value()->to(runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
    const auto *offsets = reinterpret_cast<const int32_t *>(offsets_cpu->data());
    for (size_t image_idx = 0; image_idx < input.pixel_values->size(); ++image_idx) {
        auto bound_cpu = input.image_bound->at(image_idx)->device().type() == infinilm::runtime::Device::Type::kCpu
                           ? input.image_bound->at(image_idx)
                           : input.image_bound->at(image_idx)->to(runtime, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
        const auto *bound = reinterpret_cast<const int64_t *>(bound_cpu->data());
        const size_t request_start = static_cast<size_t>(offsets[request_ids->at(image_idx)]);
        const size_t image_start = static_cast<size_t>(bound[0]);
        const size_t image_length = static_cast<size_t>(bound[1] - bound[0]);
        auto features = vision_tower_->forward(
            runtime, input.pixel_values->at(image_idx), input.image_grid_thw->at(image_idx));
        auto image_embeds = mm_projector_->forward(runtime, features);
        if (image_embeds->size(0) != image_length) {
            throw std::runtime_error("KimiK3: projected image length does not match image_bound");
        }
        inputs_embeds->narrow({{1, request_start + image_start, image_length}})
            ->copy_from(runtime, image_embeds->unsqueeze(0));
    }
}

InfinilmModel::Output KimiK3ForConditionalGeneration::forward(
    const Input &input, global_state::ForwardContext &forward_context) const {
    forward_context.require_runtime().assertOwnerThread();
    if (!input.pixel_values.has_value() || input.pixel_values->empty()
        || !language_model_->model().is_first_pp_stage()) {
        return language_model_->forward(input, forward_context);
    }
    auto inputs_embeds = language_model_->model().embed_tokens(forward_context.require_runtime(), input.input_ids.value());
    replace_image_embeddings(inputs_embeds, input, forward_context);
    auto hidden_states = language_model_->model().forward_embeds(inputs_embeds, forward_context);
    if (!language_model_->model().is_last_pp_stage()) {
        return {infinilm::runtime::Tensor(), hidden_states};
    }
    return {language_model_->logits_from_hidden(forward_context.require_runtime(), hidden_states), hidden_states};
}

void KimiK3ForConditionalGeneration::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank_info) {
    if (cache_config == nullptr) {
        return;
    }
    if (!forward_context.runtime) {
        throw std::invalid_argument("K3 reset needs explicit worker Runtime");
    }
    auto &runtime = *forward_context.runtime;
    cache::validate_cache_runtime(runtime, rank_info);
    if (owner_tp_size_ != static_cast<size_t>(rank_info.tp_size)
        || owner_pp_size_ != static_cast<size_t>(rank_info.pp_size)
        || owner_pp_stage_ != static_cast<size_t>(rank_info.pp_stage)) {
        throw std::invalid_argument("K3 cache rank topology differs from construction owner");
    }
    auto allocated = kimi_k3_allocate_cache_tensors(
        runtime, rank_info,
        cache_config,
        model_config_,
        attention_backend);
    cache_config_ = cache_config->unique_copy();
    forward_context.kv_cache_vec = std::move(allocated.kv_cache_tensors);
    forward_context.conv_state_vec = std::move(allocated.conv_state_tensors);
    forward_context.ssm_state_vec = std::move(allocated.ssm_state_tensors);
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

    // Preserve empty/all-KDA and PP configurations; guard only used denominators.
    if (model_config->get<int64_t>("hidden_size") <= 0
        || model_config->get<int64_t>("vocab_size") <= 0
        || model_config->get<int64_t>("num_hidden_layers") < 0
        || model_config->get<int64_t>("attn_res_block_size") <= 0) {
        throw std::invalid_argument("Invalid kimi_k3 construction dimensions");
    }

    const size_t num_layers = config.at("num_hidden_layers").get<size_t>();
    std::vector<std::string> layer_types(num_layers, "full_attention");
    for (const size_t one_based_layer_idx :
         config.at("linear_attn_config").at("kda_layers").get<std::vector<size_t>>()) {
        if (!one_based_layer_idx || one_based_layer_idx > num_layers) {
            throw std::invalid_argument("kimi_k3 KDA layer index is outside the one-based layer range");
        }
        layer_types.at(one_based_layer_idx - 1) = "linear_attention";
    }
    const size_t first_dense = num_layers ? model_config->get<size_t>("first_k_dense_replace") : 0;
    const size_t frequency = num_layers ? model_config->get<size_t>("moe_layer_freq") : 0;
    if (first_dense < num_layers && frequency == 0) {
        throw std::invalid_argument("kimi_k3 active MoE selection requires nonzero moe_layer_freq");
    }
    bool needs_dense = false;
    bool needs_mla = false;
    for (size_t i = 0; i < num_layers; ++i) {
        needs_dense = needs_dense || !(i >= first_dense && i % frequency == 0);
        needs_mla = needs_mla || layer_types[i] == "full_attention";
    }
    if (needs_dense && model_config->get<int64_t>("intermediate_size") <= 0) {
        throw std::invalid_argument("kimi_k3 dense layers require positive intermediate_size");
    }
    if (needs_mla) {
        for (const char *key : {"num_attention_heads", "q_lora_rank", "kv_lora_rank", "v_head_dim"}) {
            if (model_config->get<int64_t>(key) <= 0) {
                throw std::invalid_argument("kimi_k3 MLA requires positive projection dimensions");
            }
        }
        if (!config.at("head_dim").get<size_t>()) {
            throw std::invalid_argument("kimi_k3 MLA requires nonzero head dimension");
        }
    }
    config["layer_types"] = std::move(layer_types);
    return model_config;
}

} // namespace infinilm::models::kimi_k3

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    kimi_k3,
    infinilm::models::kimi_k3::KimiK3ForConditionalGeneration,
    infinilm::models::kimi_k3::create_kimi_k3_model_config);
} // namespace
