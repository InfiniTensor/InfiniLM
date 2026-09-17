#include "gemma2_for_causal_lm.hpp"
#include "../models_registry.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include "infinicore/ops/select_last_token_hidden.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::models::gemma2 {

Gemma2ForCausalLM::Gemma2ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device) {
    model_config_ = model_config;

    size_t hidden_size = model_config->get<size_t>("hidden_size");
    size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype{model_config->get_dtype()};
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    pp_size_ = static_cast<size_t>(rank_info.pp_size);
    pp_stage_ = static_cast<size_t>(rank_info.pp_stage);

    final_logit_softcapping_ = model_config->get_or<float>("final_logit_softcapping", 0.0f);

    model_ = this->register_module<Gemma2Model>("model", model_config, device);
    if (is_last_pp_stage()) {
        lm_head_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("lm_head", hidden_size, vocab_size, false, dtype, device);
    }
}

infinilm::InfinilmModel::Output Gemma2ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    if (!is_last_pp_stage()) {
        return {infinicore::Tensor(), hidden_states};
    }

    // Packed prefill only needs last-token logits for sampling; keep the
    // soft-capping passes off the full [S, vocab] tensor (same as
    // TextCausalLM).
    auto lm_head_input = hidden_states;
    if (!input.sample_all_positions && input.input_offsets.has_value()) {
        const size_t num_requests = input.input_offsets.value()->numel() - 1;
        const bool is_packed_prefill = hidden_states->ndim() == 3
                                    && hidden_states->size(0) == 1
                                    && hidden_states->size(1) > num_requests;
        if (is_packed_prefill) {
            lm_head_input = infinicore::Tensor::empty(
                {1, num_requests, hidden_states->size(2)},
                hidden_states->dtype(),
                hidden_states->device());
            infinicore::op::select_last_token_hidden_(
                lm_head_input, hidden_states, input.input_offsets.value());
        }
    }

    auto logits = lm_head_->forward(lm_head_input);
    if (final_logit_softcapping_ > 0.0f) {
        // Gemma-2 final logit soft-capping: squash logits into [-cap, cap].
        logits = infinicore::op::tanh(infinicore::op::mul_scalar(logits, 1.0f / final_logit_softcapping_));
        logits = infinicore::op::mul_scalar(logits, final_logit_softcapping_);
    }
    return {logits, hidden_states};
}

infinicore::Tensor Gemma2ForCausalLM::logits_from_hidden(const infinicore::Tensor &hidden_states) const {
    if (!lm_head_) {
        throw std::runtime_error("Gemma2ForCausalLM::logits_from_hidden called on a non-last pipeline stage");
    }
    auto logits = lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    if (final_logit_softcapping_ > 0.0f) {
        logits = infinicore::op::tanh(infinicore::op::mul_scalar(logits, 1.0f / final_logit_softcapping_));
        logits = infinicore::op::mul_scalar(logits, final_logit_softcapping_);
    }
    return logits;
}

std::shared_ptr<infinilm::config::ModelConfig> create_gemma2_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("gemma2" != model_type) {
        throw std::runtime_error(
            "infinilm::models::gemma2::create_gemma2_model_config: model_type is not gemma2");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    // Gemma-2 uses a dedicated head_dim that is NOT hidden_size / num_attention_heads
    // (e.g. 2b: hidden 2304, 8 heads, head_dim 256). Never fall back to the quotient;
    // query_pre_attn_scalar is an equal-valued but semantically distinct field.
    if (!config_json.contains("head_dim")) {
        if (config_json.contains("query_pre_attn_scalar")) {
            config_json["head_dim"] = model_config->get<size_t>("query_pre_attn_scalar");
        } else {
            throw std::runtime_error(
                "infinilm::models::gemma2::create_gemma2_model_config: config lacks head_dim and query_pre_attn_scalar");
        }
    }

    // The generic attention module defaults attention_bias to true; Gemma-2 has none.
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }

    // Gemma-2 default soft-capping values (HF Gemma2Config defaults).
    if (!config_json.contains("attn_logit_softcapping")) {
        config_json["attn_logit_softcapping"] = 50.0;
    }
    if (!config_json.contains("final_logit_softcapping")) {
        config_json["final_logit_softcapping"] = 30.0;
    }
    if (model_config->get_or<float>("attn_logit_softcapping", 50.0f) < 0.0f || model_config->get_or<float>("final_logit_softcapping", 30.0f) < 0.0f) {
        throw std::runtime_error(
            "infinilm::models::gemma2::create_gemma2_model_config: soft-capping values must be non-negative");
    }

    // Only the tanh-approximated GELU is implemented (matches the checkpoints).
    const std::string activation = config_json.value("hidden_activation", "gelu_pytorch_tanh");
    if (activation != "gelu_pytorch_tanh") {
        throw std::runtime_error(
            "infinilm::models::gemma2::create_gemma2_model_config: unsupported hidden_activation: " + activation);
    }

    // Sliding-window attention (alternating local layers) is intentionally not
    // implemented; correctness holds for sequences up to sliding_window length,
    // where the local window covers the full causal context. Warn when the
    // checkpoint's context can exceed that limit so long prompts fail loudly
    // in logs instead of silently producing wrong attention.
    if (config_json.contains("sliding_window") && config_json.contains("max_position_embeddings")) {
        const size_t sliding_window = config_json["sliding_window"].get<size_t>();
        const size_t max_position = config_json["max_position_embeddings"].get<size_t>();
        if (sliding_window < max_position) {
            spdlog::warn(
                "infinilm::models::gemma2: sliding-window attention is not implemented; results are only "
                "correct for sequences up to sliding_window={} (max_position_embeddings={})",
                sliding_window, max_position);
        }
    }
    return model_config;
}

} // namespace infinilm::models::gemma2

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    gemma2,
    infinilm::models::gemma2::Gemma2ForCausalLM,
    infinilm::models::gemma2::create_gemma2_model_config);

} // namespace
