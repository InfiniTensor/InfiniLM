#include "qwen3_next_for_causal_lm.hpp"
#include "../../global_state/forward_context.hpp"
#include "../models_registry.hpp"
#include "qwen3_next_allocate_kv_cache_tensors.hpp"
#include "runtime/runtime.hpp"
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::models::qwen3_next {

Qwen3NextForCausalLM::Qwen3NextForCausalLM(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    model_config_ = model_config;
    const size_t tp = static_cast<size_t>(rank_info.tp_size);
    const auto layer_types = model_config->get<std::vector<std::string>>("layer_types");
    for (const auto &type : layer_types) {
        if (type == "full_attention") {
            const size_t heads = model_config->get<size_t>("num_attention_heads");
            const size_t kv = model_config->get<size_t>("num_key_value_heads");
            if (heads % tp || (kv >= tp ? kv % tp : tp % kv)) {
                throw std::invalid_argument("Qwen3Next: invalid full-attention TP geometry");
            }
        } else if (model_config->get<size_t>("linear_num_key_heads") % tp
                   || model_config->get<size_t>("linear_num_value_heads") % tp) {
            throw std::invalid_argument("Qwen3Next: invalid linear-attention TP geometry");
        }
    }
    if (model_config->get<size_t>("shared_expert_intermediate_size") % tp
        || (ep_config.backend == layers::moe::EPBackend::Disabled
            && model_config->get<size_t>("moe_intermediate_size") % tp)) {
        throw std::invalid_argument("Qwen3Next: invalid expert TP geometry");
    }
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto dtype = model_config->get_dtype();
    try {
        model_ = std::make_shared<Qwen3NextModel>(
            runtime, model_config, rank_info, attention_backend, ep_config);
        this->bind_child("model", model_);
        lm_head_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
            runtime, hidden_size, vocab_size, false, dtype);
        this->bind_child("lm_head", lm_head_);
    } catch (...) {
        const auto failure = std::current_exception();
        try {
            runtime.syncStream();
        } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

infinilm::InfinilmModel::Output Qwen3NextForCausalLM::forward(
    const infinilm::InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    auto hidden_states = model_->forward(input, forward_context);
    auto logits = lm_head_->forward(*forward_context.runtime, hidden_states);
    return {logits};
}

void Qwen3NextForCausalLM::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank_info) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr, forward_context, attention_backend, rank_info);
        return;
    }
    auto &runtime = forward_context.require_runtime();
    cache::validate_cache_runtime(runtime, rank_info);
    cache_config_ = cache_config->unique_copy();

    forward_context.kv_cache_vec.clear();
    forward_context.conv_state_vec.clear();
    forward_context.ssm_state_vec.clear();

    auto cache_vectors = qwen3_next_allocate_cache_tensors(runtime, rank_info, cache_config, model_config_, attention_backend);
    forward_context.kv_cache_vec = std::move(cache_vectors.kv_cache_tensors);
    forward_context.conv_state_vec = std::move(cache_vectors.conv_state_tensors);
    forward_context.ssm_state_vec = std::move(cache_vectors.ssm_state_tensors);
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_next_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_next" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_next::create_qwen3_next_model_config: model_type is not qwen3_next");
    }
    const auto positive = [&model_config](const char *key) {
        if (model_config->get<int64_t>(key) <= 0) {
            throw std::invalid_argument(std::string("Qwen3Next requires positive ") + key);
        }
        return model_config->get<size_t>(key);
    };
    const size_t limit = std::numeric_limits<size_t>::max();
    const auto product = [limit](size_t a, size_t b) {
        if (b && a > limit / b) {
            throw std::length_error("Qwen3Next construction size overflow");
        }
        return a * b;
    };
    const size_t hidden = positive("hidden_size");
    const size_t layers = positive("num_hidden_layers");
    (void)product(hidden, positive("vocab_size"));
    const auto dtype = model_config->get_dtype();
    if (dtype != infinilm::runtime::DataType::kFloat32 && dtype != infinilm::runtime::DataType::kFloat16
        && dtype != infinilm::runtime::DataType::kBFloat16) {
        throw std::invalid_argument("Qwen3Next requires floating model dtype");
    }
    const double eps = model_config->get<double>("rms_norm_eps");
    if (!std::isfinite(eps) || eps <= 0) {
        throw std::invalid_argument("Qwen3Next requires finite positive rms_norm_eps");
    }

    nlohmann::json &config_json = model_config->get_config_json();
    if (!config_json.contains("layer_types")) {
        const size_t interval = positive("full_attention_interval");
        std::vector<std::string> layer_types;
        layer_types.reserve(layers);
        for (size_t i = 0; i < layers; i++) {
            layer_types.push_back(bool((i + 1) % interval) ? "linear_attention" : "full_attention");
        }
        config_json["layer_types"] = std::move(layer_types);
    }
    const auto layer_types = model_config->get<std::vector<std::string>>("layer_types");
    if (layer_types.size() != layers) {
        throw std::invalid_argument("Qwen3Next layer_types must match num_hidden_layers");
    }
    for (const auto &type : layer_types) {
        if (type == "full_attention") {
            const size_t heads = positive("num_attention_heads"), kv = positive("num_key_value_heads");
            const size_t dim = positive("head_dim");
            if (heads % kv || dim < 2) {
                throw std::invalid_argument("Qwen3Next full-attention head geometry is invalid");
            }
            (void)positive("max_position_embeddings");
            (void)product(product(heads, dim), hidden);
        } else if (type == "linear_attention") {
            const size_t key_heads = positive("linear_num_key_heads"), value_heads = positive("linear_num_value_heads");
            const size_t key_dim = product(key_heads, positive("linear_key_head_dim"));
            const size_t value_dim = product(value_heads, positive("linear_value_head_dim"));
            if (value_heads % key_heads || key_dim > (limit - value_dim) / 2) {
                throw std::invalid_argument("Qwen3Next linear-attention head geometry is invalid");
            }
            const int64_t kernel = model_config->get<int64_t>("linear_conv_kernel_dim");
            if (kernel < 0) {
                throw std::invalid_argument("Qwen3Next requires nonnegative linear_conv_kernel_dim");
            }
            (void)product(key_dim * 2 + value_dim, static_cast<size_t>(kernel));
            (void)product(hidden, key_dim * 2 + value_dim);
            (void)product(hidden, value_dim);
        } else {
            throw std::invalid_argument("Qwen3Next unsupported layer_type: " + type);
        }
    }
    const size_t experts = positive("num_experts"), topk = positive("num_experts_per_tok");
    if (topk > experts) {
        throw std::invalid_argument("Qwen3Next top-k exceeds num_experts");
    }
    (void)product(product(positive("moe_intermediate_size"), 2), hidden);
    (void)product(product(positive("shared_expert_intermediate_size"), 2), hidden);
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    return model_config;
}

} // namespace infinilm::models::qwen3_next

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    qwen3_next,
    infinilm::models::qwen3_next::Qwen3NextForCausalLM,
    infinilm::models::qwen3_next::create_qwen3_next_model_config);
} // namespace
