#include "minicpm_eagle_for_causal_lm.hpp"
#include "../../execution/submit.hpp"
#include "runtime/runtime.hpp"
#include "../../utils.hpp"
#include "../models_registry.hpp"


#include <exception>
#include <stdexcept>
#include <string>

namespace infinilm::models::minicpm_eagle {

MiniCPMEagleModel::MiniCPMEagleModel(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info, attention_backend, ep_config);
    try {
        dtype_ = model_config->get_dtype();
        device_ = runtime.device();
        hidden_size_ = model_config->get<size_t>("hidden_size");

    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        embed_tokens_ = this->bind_child("embed_tokens", std::make_shared<std::remove_reference_t<decltype(*embed_tokens_ )>>(runtime, vocab_size, hidden_size_, std::nullopt, dtype_));
        // Eagle/MTP-specific input projection weights. The draft model fuses the
        // previous draft token embedding with target-model hidden states through
        // input_norm1, input_norm2, and fc before running eagle_layers.
        input_norm1_ = this->bind_child("input_norm1", std::make_shared<std::remove_reference_t<decltype(*input_norm1_ )>>(runtime, hidden_size_, rms_norm_eps, dtype_));
        input_norm2_ = this->bind_child("input_norm2", std::make_shared<std::remove_reference_t<decltype(*input_norm2_ )>>(runtime, hidden_size_, rms_norm_eps, dtype_));
        fc_ = this->bind_child("fc", std::make_shared<std::remove_reference_t<decltype(*fc_ )>>(runtime, hidden_size_ * 2, hidden_size_, false, dtype_));

    eagle_layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
            eagle_layers_.push_back(this->bind_child("eagle_layers." + std::to_string(i), std::make_shared<infinilm::models::minicpm4::MiniCPM4DecoderLayer>(runtime, model_config, i, rank_info, attention_backend, ep_config)));
    }

        norm_ = this->bind_child("norm", std::make_shared<std::remove_reference_t<decltype(*norm_ )>>(runtime, hidden_size_, rms_norm_eps, dtype_));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor MiniCPMEagleModel::forward(const infinilm::InfinilmModel::Input &input,
                                          global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    infinilm::runtime::Tensor target_hidden_states;
    if (input.target_hidden_states.has_value()) {
        target_hidden_states = input.target_hidden_states.value();
    } else {
        target_hidden_states = infinilm::runtime::Tensor::empty(runtime, {input_ids->shape()[0], input_ids->shape()[1], hidden_size_}, dtype_);
        if (target_hidden_states->nbytes() != 0) {
            if (target_hidden_states->is_contiguous() && !runtime.isGraphRecording()) {
                if (!infini::ops::Fill::valid_output(infinilm::execution::detail::view(target_hidden_states)))
                    throw std::invalid_argument("zero-fill output layout or storage is invalid");
                runtime.setDeviceMemoryAsync(target_hidden_states->data(), 0, target_hidden_states->nbytes(), runtime.stream());
            } else {
                infinilm::execution::submit<infini::ops::Fill>(
                    runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
                    target_hidden_states, 0.0, target_hidden_states);
}
            INFINILM_TRACE_OP(Zeros, target_hidden_states);
        }
    }
    auto input_embeds = input_norm1_->forward(runtime, embed_tokens_->forward(runtime, input_ids));
    auto target_hidden = input_norm2_->forward(runtime, target_hidden_states);
    auto fused_shape = input_embeds->shape();
    fused_shape.back() = hidden_size_ * 2;
    auto fused_input = infinilm::runtime::Tensor::empty(runtime, fused_shape, input_embeds->dtype());
    fused_input->narrow({{fused_shape.size() - 1, 0, hidden_size_}})->copy_from(runtime, input_embeds);
    fused_input->narrow({{fused_shape.size() - 1, hidden_size_, hidden_size_}})->copy_from(runtime, target_hidden);
    auto hidden_states = fc_->forward(runtime, fused_input);

    for (const auto &layer : eagle_layers_) {
        hidden_states = layer->forward(positions, hidden_states, forward_context);
    }

    return hidden_states;
}

MiniCPMEagleForCausalLM::MiniCPMEagleForCausalLM(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info, attention_backend, ep_config);
    try {
    model_config_ = model_config;
    const auto &dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");

        model_ = this->bind_child("model", std::make_shared<std::remove_reference_t<decltype(*model_ )>>(runtime, model_config, rank_info, attention_backend, ep_config));
        lm_head_ = this->bind_child("lm_head", std::make_shared<std::remove_reference_t<decltype(*lm_head_ )>>(runtime, hidden_size, vocab_size, false, dtype));

    if (model_config->get_config_json().contains("dim_model_base")) {
        const float dim_model_base = model_config->get<float>("dim_model_base");
        lm_head_->set_alpha(dim_model_base / static_cast<float>(hidden_size));
    }
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::InfinilmModel::Output MiniCPMEagleForCausalLM::forward(
    const infinilm::InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto hidden_states = model_->forward(input, forward_context);
    auto logits = lm_head_->forward(runtime, hidden_states);
    return {logits, hidden_states};
}

std::shared_ptr<infinilm::config::ModelConfig> create_minicpm_eagle_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("minicpm_eagle" != model_type && "minicpm" != model_type) {
        throw std::runtime_error("infinilm::models::minicpm_eagle::create_minicpm_eagle_model_config: model_type is not minicpm_eagle");
    }

    auto &json = model_config->get_config_json();
    if (!json.contains("rope_theta")) {
        json["rope_theta"] = 10000.0;
    }
    if (json.contains("bias")) {
        json["attention_bias"] = json["bias"];
        json["mlp_bias"] = json["bias"];
    }
    if (!json.contains("attention_output_bias")) {
        json["attention_output_bias"] = false;
    }

    return model_config;
}

} // namespace infinilm::models::minicpm_eagle

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    minicpm_eagle,
    infinilm::models::minicpm_eagle::MiniCPMEagleForCausalLM,
    infinilm::models::minicpm_eagle::create_minicpm_eagle_model_config);
} // namespace
