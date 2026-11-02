#include "qwen3_dspark_for_causal_lm.hpp"

#include "../../engine/request_sampling.hpp"
#include "../../execution/submit.hpp"
#include "../models_registry.hpp"

#include <stdexcept>

namespace infinilm::models::qwen3_dspark {

Qwen3DSparkForCausalLM::Qwen3DSparkForCausalLM(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    if (rank_info.tp_size != 1 || rank_info.pp_size != 1) {
        throw std::invalid_argument(
            "initial Qwen3 DSpark support requires TP=1 and PP=1");
    }
    if (attention_backend == backends::AttentionBackend::STATIC_ATTN) {
        throw std::invalid_argument("Qwen3 DSpark requires a paged KV cache");
    }

    model_config_ = model_config;
    const auto dtype = model_config->get_dtype();
    hidden_size_ = model_config->get<size_t>("hidden_size");
    vocab_size_ = model_config->get<size_t>("vocab_size");
    markov_rank_ = model_config->get<size_t>("markov_rank");
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    const auto target_layer_ids = model_config->get<std::vector<size_t>>("target_layer_ids");
    const double eps = model_config->get<double>("rms_norm_eps");
    if (target_layer_ids.empty() || markov_rank_ == 0) {
        throw std::invalid_argument(
            "Qwen3 DSpark requires target_layer_ids and a positive markov_rank");
    }

    embed_tokens_ = this->bind_child(
        "embed_tokens", std::make_shared<infinilm::layers::Embedding>(
                            runtime, vocab_size_, hidden_size_, std::nullopt, dtype));
    layers_.reserve(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        layers_.push_back(this->bind_child(
            "layers." + std::to_string(layer),
            std::make_shared<Qwen3DSparkDecoderLayer>(
                runtime, model_config, layer, rank_info, attention_backend, ep_config)));
    }
    norm_ = this->bind_child(
        "norm", std::make_shared<infinilm::layers::RMSNorm>(
                    runtime, hidden_size_, eps, dtype));
    fc_ = this->bind_child(
        "fc", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
                  runtime, target_layer_ids.size() * hidden_size_, hidden_size_, false,
                  dtype));
    hidden_norm_ = this->bind_child(
        "hidden_norm", std::make_shared<infinilm::layers::RMSNorm>(
                           runtime, hidden_size_, eps, dtype));
    lm_head_ = this->bind_child(
        "lm_head", std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
                       runtime, hidden_size_, vocab_size_, false, dtype));
    markov_w1_ = this->bind_child(
        "markov_head.markov_w1",
        std::make_shared<infinilm::layers::Embedding>(
            runtime, vocab_size_, markov_rank_, std::nullopt, dtype));
    markov_w2_ = this->bind_child(
        "markov_head.markov_w2",
        std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, markov_rank_, vocab_size_, false, dtype));
    confidence_proj_ = this->bind_child(
        "confidence_head.proj",
        std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, hidden_size_ + markov_rank_, 1, true, dtype));
    if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
        runtime.syncStream();
    }
}

InfinilmModel::Output Qwen3DSparkForCausalLM::forward(
    const Input &input,
    infinilm::global_state::ForwardContext &forward_context) const {
    if (input.draft_base_logits.has_value()) {
        return forward_markov(input, forward_context);
    }
    if (input.target_hidden_states.has_value()) {
        inject_target_hidden(input, forward_context);
    }
    if (input.cache_only) {
        return {dummy_logits(input, forward_context.require_runtime()), {}};
    }

    auto &runtime = forward_context.require_runtime();
    if (!input.input_ids.has_value() || !input.position_ids.has_value()) {
        throw std::invalid_argument("Qwen3 DSpark query forward requires token and position IDs");
    }
    auto hidden_states = embed_tokens_->forward(runtime, input.input_ids.value());
    if (input.draft_anchor_tokens.has_value()) {
        const auto &anchors = input.draft_anchor_tokens.value();
        const size_t batch = anchors->numel();
        if (batch == 0 || input.input_ids.value()->numel() % batch != 0) {
            throw std::invalid_argument(
                "Qwen3 DSpark anchors do not match the packed draft query");
        }
        const size_t steps = input.input_ids.value()->numel() / batch;
        auto anchor_hidden = embed_tokens_->forward(
            runtime, anchors->view({1, batch}));
        hidden_states->view({batch, steps, hidden_size_})
            ->narrow({{1, 0, 1}})
            ->copy_from(
                runtime, anchor_hidden->view({batch, 1, hidden_size_}));
    }
    infinilm::runtime::Tensor residual;
    for (const auto &layer : layers_) {
        layer->forward(input.position_ids.value(), hidden_states, residual, forward_context);
    }
    norm_->forward_inplace(runtime, hidden_states, residual);
    auto lm_input = hidden_states;
    auto logits = lm_head_->forward(runtime, lm_input);
    if (input.draft_anchor_tokens.has_value()) {
        return apply_markov(
            logits, input.draft_anchor_tokens.value(), runtime);
    }
    return {logits, hidden_states};
}

InfinilmModel::Output Qwen3DSparkForCausalLM::forward_markov(
    const Input &input,
    infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    if (!input.input_ids.has_value()) {
        throw std::invalid_argument("Qwen3 DSpark Markov step requires previous tokens");
    }
    return apply_markov(
        input.draft_base_logits.value(), input.input_ids.value(),
        forward_context.require_runtime());
}

InfinilmModel::Output Qwen3DSparkForCausalLM::apply_markov(
    const infinilm::runtime::Tensor &base,
    const infinilm::runtime::Tensor &previous_tokens,
    infinilm::runtime::Runtime &runtime) const {
    if (base->ndim() != 3 || base->size(0) != 1 || previous_tokens->numel() == 0
        || base->size(1) % previous_tokens->numel() != 0) {
        throw std::invalid_argument(
            "Qwen3 DSpark Markov inputs require [1, batch * steps, vocab] "
            "base logits and one previous token per batch row");
    }

    const size_t batch = previous_tokens->numel();
    const size_t steps = base->size(1) / batch;
    const size_t vocab = base->size(2);
    auto logits = infinilm::runtime::Tensor::empty(runtime, base->shape(), base->dtype());
    auto base_rows = base->as_strided(
        {batch, steps, vocab},
        {static_cast<ptrdiff_t>(steps * vocab), static_cast<ptrdiff_t>(vocab), 1});
    auto output_rows = logits->as_strided(
        {batch, steps, vocab},
        {static_cast<ptrdiff_t>(steps * vocab), static_cast<ptrdiff_t>(vocab), 1});
    auto current_tokens = previous_tokens->view({batch});
    infinilm::runtime::Tensor embedding;

    for (size_t step = 0; step < steps; ++step) {
        embedding = markov_w1_->forward(runtime, current_tokens);
        auto bias_input = embedding;
        auto bias = markov_w2_->forward(runtime, bias_input);
        auto base_step = base_rows->narrow({{1, step, 1}})->as_strided({batch, vocab}, {static_cast<ptrdiff_t>(steps * vocab), 1});
        auto output_step = output_rows->narrow({{1, step, 1}})->as_strided({batch, vocab}, {static_cast<ptrdiff_t>(steps * vocab), 1});
        if (bias->shape() != output_step->shape()) {
            throw std::invalid_argument(
                "Qwen3 DSpark Markov bias does not match a base-logit step");
        }
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infinilm::execution::RecordingPolicy::AllowHostReplay,
            infini::ops::Add::DefaultConfig(runtime.device().type()),
            base_step, bias, output_step);
        INFINILM_TRACE_OP(Add, output_step);

        if (step + 1 < steps) {
            current_tokens = infinilm::runtime::Tensor::empty(
                runtime, {batch}, infinilm::runtime::DataType::kInt64);
            infinilm::engine::sample_request_token(
                runtime, current_tokens, output_step, 0.0f, 1.0f, 1, 1.0f);
        }
    }
    return {logits, embedding};
}

void Qwen3DSparkForCausalLM::inject_target_hidden(
    const Input &input,
    infinilm::global_state::ForwardContext &forward_context) const {
    if (!input.position_ids.has_value() || !input.slot_mapping.has_value()) {
        throw std::invalid_argument(
            "Qwen3 DSpark cache injection requires positions and slot mapping");
    }
    auto &runtime = forward_context.require_runtime();
    auto projection_input = input.target_hidden_states.value();
    auto projected = fc_->forward(runtime, projection_input);
    projected = hidden_norm_->forward(runtime, projected);
    const auto &positions = input.target_hidden_position_ids.has_value()
                              ? input.target_hidden_position_ids.value()
                              : input.position_ids.value();
    const auto &slot_mapping = input.target_hidden_slot_mapping.has_value()
                                 ? input.target_hidden_slot_mapping.value()
                                 : input.slot_mapping.value();
    for (const auto &layer : layers_) {
        layer->inject_target_hidden(
            positions, projected, slot_mapping,
            forward_context);
    }
}

infinilm::runtime::Tensor Qwen3DSparkForCausalLM::dummy_logits(
    const Input &input,
    infinilm::runtime::Runtime &runtime) const {
    if (!input.input_offsets.has_value() || input.input_offsets.value()->numel() < 2) {
        throw std::invalid_argument(
            "Qwen3 DSpark cache-only forward requires input offsets");
    }
    const size_t requests = input.input_offsets.value()->numel() - 1;
    return infinilm::runtime::Tensor::empty(
        runtime, {1, requests, 1}, model_config_->get_dtype());
}

std::shared_ptr<infinilm::config::ModelConfig>
create_qwen3_dspark_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (!model_config
        || model_config->get<std::string>("model_type") != "qwen3_dspark") {
        throw std::invalid_argument("Qwen3 DSpark configuration is required");
    }
    models::validate_dense_construction_metadata(model_config, "Qwen3 DSpark");
    return model_config;
}

} // namespace infinilm::models::qwen3_dspark

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    qwen3_dspark,
    infinilm::models::qwen3_dspark::Qwen3DSparkForCausalLM,
    infinilm::models::qwen3_dspark::create_qwen3_dspark_model_config);
} // namespace
