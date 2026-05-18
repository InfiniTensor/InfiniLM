#include "minicpmv_model.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "infinicore/ops.hpp"

#include <algorithm>
#include <stdexcept>

namespace infinilm::models::minicpmv {

MiniCPMVModel::MiniCPMVModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             const infinicore::Device &device)
    : config_(model_config) {

    const auto &dtype{model_config->get_dtype()};

    INFINICORE_NN_MODULE_INIT(llm, model_config, device);

    // Use get_ref instead of get to get a reference
    auto &vision_cfg = model_config->get_ref("vision_config");

    if (model_config->get<bool>("drop_vision_last_layer") && vision_cfg.value("num_hidden_layers", 0) > 0) {
        vision_cfg["num_hidden_layers"] = vision_cfg.value("num_hidden_layers", 0) - 1;
    }

    INFINICORE_NN_MODULE_INIT(vpm, vision_cfg, dtype, device, false);

    size_t embed_dim = model_config->get<size_t>("hidden_size");
    size_t num_heads = embed_dim / 128;
    INFINICORE_NN_MODULE_INIT(resampler,
                              model_config->get<size_t>("query_num"),
                              embed_dim,
                              num_heads,
                              vision_cfg.value("hidden_size", 768),
                              dtype,
                              device);
}

infinicore::Tensor MiniCPMVModel::replace_embeddings(const infinicore::Tensor &inputs_embeds,
                                                     const infinicore::Tensor &vision_hidden,
                                                     const infinicore::Tensor &image_bound) const {
    auto out = infinicore::Tensor::empty(inputs_embeds->shape(), inputs_embeds->dtype(), inputs_embeds->device());
    out->copy_from(inputs_embeds);

    auto bounds_cpu = image_bound->to(infinicore::Device::cpu());
    auto batch_size = inputs_embeds->size(0);

    ASSERT_EQ(batch_size, 1);
    ASSERT_EQ(bounds_cpu->size(0), 1);
    auto out_slice = out->squeeze(0);
    auto bound_slice = bounds_cpu->squeeze(0);
    auto vision_len = vision_hidden->size(0);
    for (size_t patch = 0; patch < vision_len; ++patch) {
        auto patch_embed = vision_hidden->narrow({{0, patch, 1}})->squeeze(0);
        auto bound = bound_slice->narrow({{0, patch, 1}});
        auto bound_ptr = reinterpret_cast<const int64_t *>(bound->data());
        auto start = bound_ptr[0];
        auto end = bound_ptr[1];

        out_slice->narrow({{0, size_t(start), size_t(end - start)}})->copy_from(patch_embed);
    }

    return out;
}

InfinilmModel::Output MiniCPMVModel::forward(const InfinilmModel::Input &input) const {
    if (!input.input_ids.has_value()) {
        throw std::runtime_error("MiniCPMVModel: input_ids is required");
    }
    auto input_ids = input.input_ids.value();

    if (input.pixel_values.has_value() && input.pixel_values.value().size() > 0) {
        if (!input.image_bound.has_value() or !input.tgt_sizes.has_value()) {
            throw std::runtime_error("MiniCPMVModel: image_bound and tgt_sizes must be provided with pixel_values");
        }
        if (input.pixel_values->size() != input.image_bound->size() || input.pixel_values->size() != input.tgt_sizes->size()) {
            throw std::runtime_error("MiniCPMVModel: pixel_values, image_bound and tgt_sizes must have the same number of elements");
        }

        auto inputs_embeds = llm_->model().embed_tokens(input_ids);

        // inputs_embeds concat tokens from all requests, while images are processed per request
        // slice inputs_embeds using request offsets to get the embedding of each request
        infinicore::Tensor input_offsets_cpu = input.input_offsets.value()->to(infinicore::Device::cpu());
        int32_t *offsets = (int32_t *)(input_offsets_cpu->data());
        for (size_t i : global_state::get_forward_context().mm_metadata.image_req_ids.value()) {
            auto pixel_values = input.pixel_values.value().at(i);
            auto vision_embedding = vpm_->forward(pixel_values, input.tgt_sizes.value().at(i));
            auto vision_hidden = resampler_->forward(vision_embedding, input.tgt_sizes.value().at(i));
            inputs_embeds = replace_embeddings(inputs_embeds->narrow({{1, size_t(offsets[i]), size_t(offsets[i + 1] - offsets[i])}}), vision_hidden, input.image_bound.value().at(i));
        }

        auto hidden_states = llm_->model().forward_embeds(
            inputs_embeds,
            input.position_ids.value());

        auto logits = llm_->logits_from_hidden(hidden_states);
        return {logits};
    }

    return llm_->forward(input);
}

void MiniCPMVModel::reset_cache(const cache::CacheConfig *cache_config) {
    llm_->reset_cache(cache_config);
}

std::shared_ptr<infinilm::config::ModelConfig> create_minicpmv_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minicpmv" != model_type) {
        throw std::runtime_error("infinilm::models::minicpmv::create_minicpmv_model_config: model_type is not minicpmv");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    if (!config_json.contains("head_dim")) {
        size_t head_dim = model_config->get<size_t>("hidden_size")
                        / model_config->get<size_t>("num_attention_heads");
        config_json["head_dim"] = head_dim;
    }

    return model_config;
}

} // namespace infinilm::models::minicpmv

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minicpmv,
    infinilm::models::minicpmv::MiniCPMVModel,
    infinilm::models::minicpmv::create_minicpmv_model_config);
} // namespace
