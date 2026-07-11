#include "ernie4_5_for_conditional_generation.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

namespace {

size_t first_size_t_or(const nlohmann::json &config_json,
                       const std::string &key,
                       size_t default_value) {
    if (!config_json.contains(key) || config_json.at(key).is_null()) {
        return default_value;
    }
    const auto &value = config_json.at(key);
    if (value.is_array()) {
        if (value.empty()) {
            return default_value;
        }
        return value.at(0).get<size_t>();
    }
    return value.get<size_t>();
}

size_t second_size_t_or(const nlohmann::json &config_json,
                        const std::string &key,
                        size_t default_value) {
    if (!config_json.contains(key) || config_json.at(key).is_null()) {
        return default_value;
    }
    const auto &value = config_json.at(key);
    if (value.is_array()) {
        if (value.size() < 2) {
            return first_size_t_or(config_json, key, default_value);
        }
        return value.at(1).get<size_t>();
    }
    return value.get<size_t>();
}

void set_if_missing(nlohmann::json &config_json,
                    const std::string &key,
                    const nlohmann::json &value) {
    if (!config_json.contains(key) || config_json.at(key).is_null()) {
        config_json[key] = value;
    }
}

void append_visual_ranges(std::vector<size_t> &ranges,
                          const infinicore::Tensor &image_bound,
                          const infinicore::Tensor &input_ids,
                          const nlohmann::json &config_json,
                          size_t token_offset) {
    auto bounds_cpu = image_bound->to(infinicore::Device::cpu());
    auto bound_slice = bounds_cpu;
    if (bound_slice->shape().size() == 3) {
        ASSERT_EQ(bound_slice->size(0), 1);
        bound_slice = bound_slice->squeeze(0);
    }

    auto ids_cpu = input_ids->to(infinicore::Device::cpu())->contiguous();
    auto ids_shape = ids_cpu->shape();
    const size_t seq_len = ids_shape.empty() ? 0 : ids_shape.back();
    std::vector<int64_t> boundary_token_ids;
    for (const auto *key : {"image_start_token_id", "image_end_token_id", "video_start_token_id", "video_end_token_id"}) {
        if (config_json.contains(key) && !config_json.at(key).is_null()) {
            boundary_token_ids.push_back(config_json.at(key).get<int64_t>());
        }
    }

    auto is_boundary_token = [&boundary_token_ids](int64_t token_id) {
        return std::find(boundary_token_ids.begin(), boundary_token_ids.end(), token_id) != boundary_token_ids.end();
    };

    auto token_at = [&ids_cpu](size_t index) -> int64_t {
        if (ids_cpu->dtype() == infinicore::DataType::I64) {
            const auto *ids = reinterpret_cast<const int64_t *>(ids_cpu->data());
            return ids[index];
        }
        if (ids_cpu->dtype() == infinicore::DataType::I32) {
            const auto *ids = reinterpret_cast<const int32_t *>(ids_cpu->data());
            return ids[index];
        }
        throw std::runtime_error("Ernie4_5ForConditionalGeneration: input_ids must be int32 or int64");
    };

    for (size_t i = 0; i < bound_slice->size(0); ++i) {
        auto bound = bound_slice->narrow({{0, i, 1}})->squeeze(0);
        const auto *bound_ptr = reinterpret_cast<const int64_t *>(bound->data());
        const int64_t start = bound_ptr[0];
        const int64_t end = bound_ptr[1];
        if (end > start) {
            size_t range_start = static_cast<size_t>(start);
            size_t range_end = static_cast<size_t>(end);
            if (range_start > 0 && range_start - 1 < seq_len && is_boundary_token(token_at(range_start - 1))) {
                --range_start;
            }
            if (range_end < seq_len && is_boundary_token(token_at(range_end))) {
                ++range_end;
            }
            ranges.push_back(token_offset + range_start);
            ranges.push_back(token_offset + range_end);
        }
    }
}

infinicore::Tensor expand_grid_for_vision_tower(const infinicore::Tensor &grid_thw) {
    auto grid_cpu = grid_thw->to(infinicore::Device::cpu())->contiguous();
    auto shape = grid_cpu->shape();
    if (shape.size() != 2 || shape[1] != 3) {
        throw std::runtime_error("Ernie4_5ForConditionalGeneration: grid_thw must have shape [num_items, 3]");
    }

    std::vector<int64_t> expanded;
    expanded.reserve(shape[0] * 3);
    const auto append_grid = [&expanded](int64_t t, int64_t h, int64_t w) {
        for (int64_t ti = 0; ti < t; ++ti) {
            (void)ti;
            expanded.push_back(1);
            expanded.push_back(h);
            expanded.push_back(w);
        }
    };

    if (grid_cpu->dtype() == infinicore::DataType::I64) {
        const auto *ptr = reinterpret_cast<const int64_t *>(grid_cpu->data());
        for (size_t i = 0; i < shape[0]; ++i) {
            append_grid(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
        }
    } else if (grid_cpu->dtype() == infinicore::DataType::I32) {
        const auto *ptr = reinterpret_cast<const int32_t *>(grid_cpu->data());
        for (size_t i = 0; i < shape[0]; ++i) {
            append_grid(ptr[i * 3 + 0], ptr[i * 3 + 1], ptr[i * 3 + 2]);
        }
    } else {
        throw std::runtime_error("Ernie4_5ForConditionalGeneration: grid_thw must be int32 or int64");
    }

    auto expanded_cpu = infinicore::Tensor::from_blob(
        expanded.data(),
        {expanded.size() / 3, 3},
        infinicore::DataType::I64,
        infinicore::Device::cpu());
    return expanded_cpu->to(grid_thw->device());
}

} // namespace

Ernie4_5ForConditionalGeneration::Ernie4_5ForConditionalGeneration(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &config_json = model_config->get_config_json();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    if (config_json.contains("vision_config")) {
        INFINICORE_NN_MODULE_INIT(vision_model,
                                  config_json.at("vision_config"),
                                  model_config->get_or<double>("rms_norm_eps", 1e-6),
                                  dtype,
                                  device);
        INFINICORE_NN_MODULE_INIT(resampler_model, config_json, dtype, device);
    }
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinicore::Tensor Ernie4_5ForConditionalGeneration::replace_embeddings(
    const infinicore::Tensor &inputs_embeds,
    const infinicore::Tensor &vision_hidden,
    const infinicore::Tensor &image_bound) const {
    auto out = infinicore::Tensor::empty(inputs_embeds->shape(), inputs_embeds->dtype(), inputs_embeds->device());
    out->copy_from(inputs_embeds);

    auto bounds_cpu = image_bound->to(infinicore::Device::cpu());
    const size_t batch_size = inputs_embeds->size(0);
    ASSERT_EQ(batch_size, 1);

    auto out_slice = out->squeeze(0);
    auto bound_slice = bounds_cpu;
    if (bound_slice->shape().size() == 3) {
        ASSERT_EQ(bound_slice->size(0), 1);
        bound_slice = bound_slice->squeeze(0);
    }

    const size_t num_ranges = bound_slice->size(0);
    const size_t vision_len = vision_hidden->size(0);
    size_t vision_offset = 0;
    for (size_t i = 0; i < num_ranges; ++i) {
        auto bound = bound_slice->narrow({{0, i, 1}})->squeeze(0);
        auto *bound_ptr = reinterpret_cast<const int64_t *>(bound->data());
        const int64_t start = bound_ptr[0];
        const int64_t end = bound_ptr[1];
        if (end <= start) {
            continue;
        }
        const size_t len = static_cast<size_t>(end - start);
        if (vision_offset + len > vision_len) {
            throw std::runtime_error("Ernie4_5ForConditionalGeneration: image_bound exceeds vision hidden length");
        }
        auto patch_embed = vision_hidden->narrow({{0, vision_offset, len}});
        out_slice->narrow({{0, static_cast<size_t>(start), len}})->copy_from(patch_embed);
        vision_offset += len;
    }
    if (vision_offset != vision_len) {
        throw std::runtime_error("Ernie4_5ForConditionalGeneration: unused vision hidden tokens after image_bound replacement");
    }
    return out;
}

infinilm::InfinilmModel::Output Ernie4_5ForConditionalGeneration::forward(
    const infinilm::InfinilmModel::Input &input) const {
    auto &mm_metadata = global_state::get_forward_context().mm_metadata;
    mm_metadata.visual_token_ranges.reset();
    if (input.pixel_values.has_value() && !input.pixel_values->empty()) {
        mm_metadata.visual_token_ranges = std::vector<size_t>{};
        if (!input.input_ids.has_value()) {
            throw std::runtime_error("Ernie4_5ForConditionalGeneration: input_ids is required");
        }
        if (!input.position_ids.has_value()) {
            throw std::runtime_error("Ernie4_5ForConditionalGeneration: position_ids is required");
        }
        if (!input.tgt_sizes.has_value()) {
            throw std::runtime_error("Ernie4_5ForConditionalGeneration: tgt_sizes/image_grid_thw is required for vision input");
        }
        if (!input.image_bound.has_value()) {
            throw std::runtime_error("Ernie4_5ForConditionalGeneration: image_bound is required for vision input");
        }
        if (input.pixel_values->size() != input.image_bound->size() || input.pixel_values->size() != input.tgt_sizes->size()) {
            throw std::runtime_error("Ernie4_5ForConditionalGeneration: pixel_values, image_bound and tgt_sizes must have the same number of elements");
        }

        const auto &config_json = model_config_->get_config_json();
        auto input_ids = input.input_ids.value();
        auto inputs_embeds = model_->embed_tokens(input_ids);

        const auto &pixel_values = input.pixel_values.value();
        const auto &image_bound = input.image_bound.value();
        const auto &tgt_sizes = input.tgt_sizes.value();

        if (input.input_offsets.has_value() && mm_metadata.image_req_ids.has_value()) {
            auto input_offsets_cpu = input.input_offsets.value()->to(infinicore::Device::cpu());
            const auto *offsets = reinterpret_cast<const int32_t *>(input_offsets_cpu->data());
            const auto &image_req_ids = mm_metadata.image_req_ids.value();
            if (image_req_ids.size() != pixel_values.size()) {
                throw std::runtime_error("Ernie4_5ForConditionalGeneration: image_req_ids size does not match pixel_values");
            }
            for (size_t image_idx = 0; image_idx < pixel_values.size(); ++image_idx) {
                const size_t req_id = image_req_ids.at(image_idx);
                auto vision_grid = expand_grid_for_vision_tower(tgt_sizes.at(image_idx));
                auto vision_hidden = vision_model_->forward(pixel_values.at(image_idx), vision_grid);
                vision_hidden = resampler_model_->forward(vision_hidden, tgt_sizes.at(image_idx));
                auto request_embeds = inputs_embeds->narrow(
                    {{1, static_cast<size_t>(offsets[req_id]), static_cast<size_t>(offsets[req_id + 1] - offsets[req_id])}});
                auto replaced = replace_embeddings(request_embeds, vision_hidden, image_bound.at(image_idx));
                request_embeds->copy_from(replaced);
                auto request_input_ids = input_ids->narrow(
                    {{1, static_cast<size_t>(offsets[req_id]), static_cast<size_t>(offsets[req_id + 1] - offsets[req_id])}});
                append_visual_ranges(
                    mm_metadata.visual_token_ranges.value(),
                    image_bound.at(image_idx),
                    request_input_ids,
                    config_json,
                    static_cast<size_t>(offsets[req_id]));
            }
        } else {
            if (pixel_values.size() != 1) {
                throw std::runtime_error("Ernie4_5ForConditionalGeneration: batched vision input requires image_req_ids");
            }
            auto vision_grid = expand_grid_for_vision_tower(tgt_sizes.at(0));
            auto vision_hidden = vision_model_->forward(pixel_values.at(0), vision_grid);
            vision_hidden = resampler_model_->forward(vision_hidden, tgt_sizes.at(0));
            inputs_embeds = replace_embeddings(inputs_embeds, vision_hidden, image_bound.at(0));
            append_visual_ranges(mm_metadata.visual_token_ranges.value(), image_bound.at(0), input_ids, config_json, 0);
        }

        auto hidden_states = model_->forward_embeds(inputs_embeds, input.position_ids.value());
        mm_metadata.visual_token_ranges.reset();
        return {lm_head_->forward(hidden_states)};
    }
    auto hidden_states = model_->forward(input);
    return {lm_head_->forward(hidden_states)};
}

std::shared_ptr<infinilm::config::ModelConfig> create_ernie4_5_moe_vl_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("ernie4_5_moe_vl" != model_type && "ernie4_5_vl_moe" != model_type) {
        throw std::runtime_error("create_ernie4_5_moe_vl_model_config: unsupported model_type: " + model_type);
    }

    nlohmann::json &config_json = model_config->get_config_json();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_attention_heads = model_config->get<size_t>("num_attention_heads");

    set_if_missing(config_json, "head_dim", hidden_size / num_attention_heads);
    set_if_missing(config_json, "attention_bias", false);
    set_if_missing(config_json, "attention_output_bias", false);
    set_if_missing(config_json, "mlp_bias", false);
    set_if_missing(config_json, "use_moe", true);

    set_if_missing(config_json, "vision_num_experts", second_size_t_or(config_json, "moe_num_experts", 0));
    set_if_missing(
        config_json,
        "vision_moe_intermediate_size",
        second_size_t_or(config_json, "moe_intermediate_size", model_config->get<size_t>("intermediate_size")));
    set_if_missing(config_json, "num_experts", first_size_t_or(config_json, "moe_num_experts", 0));
    set_if_missing(config_json, "num_experts_per_tok", first_size_t_or(config_json, "moe_k", 1));
    config_json["moe_intermediate_size"] = first_size_t_or(config_json, "moe_intermediate_size", model_config->get<size_t>("intermediate_size"));
    set_if_missing(config_json, "norm_topk_prob", true);
    set_if_missing(config_json, "moe_router_backend", "softmax");
    set_if_missing(config_json, "moe_router_dtype", "float32");
    set_if_missing(config_json, "moe_router_use_correction_bias", true);
    set_if_missing(config_json, "enable_ernie_vision_experts", true);

    model_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);

    return model_config;
}

} // namespace infinilm::models::ernie4_5_moe_vl

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    ernie4_5_moe_vl,
    infinilm::models::ernie4_5_moe_vl::Ernie4_5ForConditionalGeneration,
    infinilm::models::ernie4_5_moe_vl::create_ernie4_5_moe_vl_model_config);
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    ernie4_5_vl_moe,
    infinilm::models::ernie4_5_moe_vl::Ernie4_5ForConditionalGeneration,
    infinilm::models::ernie4_5_moe_vl::create_ernie4_5_moe_vl_model_config);
} // namespace
