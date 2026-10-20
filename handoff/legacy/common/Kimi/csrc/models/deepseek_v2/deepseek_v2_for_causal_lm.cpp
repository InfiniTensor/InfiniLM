#include "../../layers/rotary_embedding/rope.hpp"
#include "deepseek_v2_for_causal_lm.hpp"
#include "runtime/runtime.hpp"
#include <exception>

#include "../models_registry.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v2 {

DeepseekV2Model::DeepseekV2Model(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config,
    bool use_mla) {
    models::validate_runtime_construction(model_config, runtime, rank_info, attention_backend, ep_config);
    if (use_mla && model_config->get_or<size_t>("q_lora_rank", 0) != 0)
        throw std::runtime_error("DeepseekV2MLAAttention: q_lora_rank is not supported yet");
    try {
    const auto &dtype{model_config->get_dtype()};
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        embed_tokens_ = this->bind_child("embed_tokens", std::make_shared<std::remove_reference_t<decltype(*embed_tokens_ )>>(runtime, vocab_size, hidden_size, std::nullopt, dtype));
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
            layers_.push_back(std::make_shared<DeepseekV2DecoderLayer>(
                runtime, model_config, i, rank_info, attention_backend, ep_config, use_mla));
            this->bind_child("layers." + std::to_string(i), layers_.back());
    }
        norm_ = this->bind_child("norm", std::make_shared<std::remove_reference_t<decltype(*norm_ )>>(runtime, hidden_size, rms_norm_eps, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor DeepseekV2Model::forward(const infinilm::InfinilmModel::Input &input,
                                                       infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = embed_tokens_->forward(runtime, input_ids);

    infinilm::runtime::Tensor residual;
    for (const auto &layer : layers_) {
        layer->forward(positions, hidden_states, residual, forward_context);
    }
    norm_->forward_inplace(runtime, hidden_states, residual);
    return hidden_states;
}

DeepseekV2ForCausalLM::DeepseekV2ForCausalLM(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config,
    bool use_mla)
    : use_mla_(use_mla) {
    models::validate_runtime_construction(model_config, runtime, rank_info, attention_backend, ep_config);
    if (use_mla && model_config->get_or<size_t>("q_lora_rank", 0) != 0)
        throw std::runtime_error("DeepseekV2MLAAttention: q_lora_rank is not supported yet");
    try {
    model_config_ = model_config;
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
        model_ = this->bind_child("model", std::make_shared<std::remove_reference_t<decltype(*model_ )>>(runtime, model_config, rank_info, attention_backend, ep_config, use_mla));
        lm_head_ = this->bind_child("lm_head", std::make_shared<std::remove_reference_t<decltype(*lm_head_ )>>(runtime, hidden_size, vocab_size, false, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::InfinilmModel::Output DeepseekV2ForCausalLM::forward(
    const infinilm::InfinilmModel::Input &input, infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto hidden_states = model_->forward(input, forward_context);
    auto logits = lm_head_->forward(runtime, hidden_states);
    return {logits};
}

void DeepseekV2ForCausalLM::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    if (!use_mla_ || cache_config == nullptr) {
        InfinilmModel::reset_cache(cache_config, forward_context, attention_backend, rank);
        return;
    }

    auto &runtime = forward_context.require_runtime();
    cache::validate_cache_runtime(runtime, rank);
    cache_config_ = cache_config->unique_copy();

    auto &kv_cache_vec = forward_context.kv_cache_vec;
    kv_cache_vec.clear();
    auto new_kv_cache_vec = deepseek_v2_allocate_kv_cache_tensors(runtime, rank, cache_config, model_config_, attention_backend);
    kv_cache_vec = std::move(new_kv_cache_vec);
}

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (!model_config) throw std::invalid_argument("DeepseekV2 requires configuration");
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("deepseek_v2" != model_type) {
        throw std::runtime_error("create_deepseek_v2_model_config: model_type is not deepseek_v2");
    }

    auto &config_json = model_config->get_config_json();
    for (const char *key : {"hidden_size", "vocab_size", "num_attention_heads", "kv_lora_rank", "qk_rope_head_dim", "v_head_dim", "intermediate_size", "max_position_embeddings"})
        if (model_config->get<int64_t>(key) <= 0) throw std::invalid_argument("DeepseekV2 requires positive construction dimensions");
    if (model_config->get<int64_t>("num_hidden_layers") < 0 || model_config->get<int64_t>("qk_nope_head_dim") < 0)
        throw std::invalid_argument("DeepseekV2 layer/nonrotary dimensions are invalid");
    const size_t nope = config_json.at("qk_nope_head_dim").get<size_t>(), rope = config_json.at("qk_rope_head_dim").get<size_t>();
    if (nope > std::numeric_limits<size_t>::max() - rope || rope % 2)
        throw std::invalid_argument("DeepseekV2 RoPE geometry is invalid");
    const size_t q_head_dim = nope + rope;
    const size_t limit = std::numeric_limits<size_t>::max(), heads = model_config->get<size_t>("num_attention_heads");
    const size_t latent = model_config->get<size_t>("kv_lora_rank"), value = model_config->get<size_t>("v_head_dim");
    if (latent > limit - rope || nope > limit - value
        || heads > limit / q_head_dim || heads > limit / (nope + value) || heads > limit / value)
        throw std::length_error("DeepseekV2 projection size overflow");
    config_json["head_dim"] = q_head_dim;
    
    const size_t qk_rope_head_dim = config_json.at("qk_rope_head_dim").get<size_t>();
    config_json["partial_rotary_factor"] = static_cast<double>(qk_rope_head_dim) / static_cast<double>(q_head_dim);
    
    config_json["num_experts"] = config_json.value("n_routed_experts", 0);
    config_json["mlp_bias"] = false;
    if (!config_json.contains("attention_output_bias")) {
        config_json["attention_output_bias"] = config_json.value("attention_bias", false);
    }
    if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
        config_json["dtype"] = config_json["torch_dtype"];
    }

    // Use GPT-J style for DeepseekV2
    model_config->set_rope_algo(infinilm::layers::RoPE::Algo::GPT_J);

    const double eps = model_config->get<double>("rms_norm_eps");
    if (!std::isfinite(eps) || eps <= 0) throw std::invalid_argument("DeepseekV2 norm epsilon is invalid");
    if (model_config->get_or<int64_t>("n_routed_experts", 0) < 0 || model_config->get_or<int64_t>("n_shared_experts", 0) < 0)
        throw std::invalid_argument("Deepseek expert counts must be nonnegative");
    const size_t routed = model_config->get_or<size_t>("n_routed_experts", 0);
    if (routed) {
        const size_t topk = model_config->get<size_t>("num_experts_per_tok");
        if (!topk || topk > routed || model_config->get<int64_t>("moe_intermediate_size") <= 0)
            throw std::invalid_argument("DeepseekV2 expert/topk dimensions are invalid");
        const size_t shared = model_config->get_or<size_t>("n_shared_experts", 0);
        if (shared && model_config->get<size_t>("moe_intermediate_size") > std::numeric_limits<size_t>::max() / shared)
            throw std::length_error("DeepseekV2 shared expert size overflow");
    }
    return model_config;
}

} // namespace infinilm::models::deepseek_v2

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    deepseek_v2,
    infinilm::models::deepseek_v2::DeepseekV2ForCausalLM,
    infinilm::models::deepseek_v2::create_deepseek_v2_model_config);
} // namespace
