#include "qwen3_5_for_causal_lm.hpp"

#include "../models_registry.hpp"
#include "runtime/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5 {

Qwen35ForCausalLM::Qwen35ForCausalLM(infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    try {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype = model_config->get_dtype();

        model_ = this->bind_child("model", std::make_shared<std::remove_reference_t<decltype(*model_ )>>(runtime, model_config, rank_info, attention_backend, ep_config));
        lm_head_ = this->bind_child("lm_head", std::make_shared<std::remove_reference_t<decltype(*lm_head_ )>>(runtime, hidden_size, vocab_size, false, dtype));
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

InfinilmModel::Output Qwen35ForCausalLM::forward(
    const InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    if (forward_context.runtime == nullptr) {
        throw std::invalid_argument("Qwen3.5 forward requires an explicit Runtime");
    }
    auto hidden_states = model_->forward(input, forward_context);
    return {lm_head_->forward(*forward_context.runtime, hidden_states)};
}

void Qwen35ForCausalLM::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank_info) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    if (cache_config != nullptr) {
        cache::validate_cache_runtime(forward_context.require_runtime(), rank_info);
    }
    if (cache_config == nullptr) {
        cache_config_.reset();
    } else {
        cache_config_ = cache_config->unique_copy();
    }
    model_->reset_cache(cache_config, forward_context, attention_backend, rank_info);
}

std::shared_ptr<infinilm::config::ModelConfig> prepare_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (!model_config) throw std::invalid_argument("Qwen3.5 requires configuration");
    nlohmann::json &config_json = model_config->get_config_json();
    if (config_json.contains("text_config") && config_json["text_config"].is_object()) {
        const nlohmann::json &text_config_json = config_json["text_config"];
        for (auto it = text_config_json.begin(); it != text_config_json.end(); ++it) {
            if (!config_json.contains(it.key())) {
                config_json[it.key()] = it.value();
            }
        }
        if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
            config_json["dtype"] = config_json["torch_dtype"];
        }
    }
    if (!config_json.contains("position_id_axes")) {
        size_t position_id_axes = 1;
        if (config_json.contains("rope_parameters")
            && config_json["rope_parameters"].is_object()) {
            const auto &rope_parameters = config_json["rope_parameters"];
            if (rope_parameters.contains("mrope_section")
                && rope_parameters["mrope_section"].is_array()
                && !rope_parameters["mrope_section"].empty()) {
                position_id_axes = rope_parameters["mrope_section"].size();
            }
        }
        config_json["position_id_axes"] = position_id_axes;
    }
    if (!config_json.contains("rope_theta") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("rope_theta")) {
        // Normalize the nested HuggingFace field for the Qwen3.5 attention module.
        config_json["rope_theta"] = config_json["rope_parameters"]["rope_theta"];
    }
    if (!config_json.contains("partial_rotary_factor") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("partial_rotary_factor")) {
        config_json["partial_rotary_factor"] = config_json["rope_parameters"]["partial_rotary_factor"];
    }
    const auto layers_signed = model_config->get<int64_t>("num_hidden_layers");
    if (layers_signed < 0) throw std::invalid_argument("Qwen3.5 requires nonnegative layer count");
    if (!config_json.contains("layer_types")) {
        const size_t full_attention_interval = model_config->get<size_t>("full_attention_interval");
        if (full_attention_interval == 0) {
            throw std::runtime_error("Qwen3.5 full_attention_interval must be positive");
        }
        const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            layer_types.push_back(
                (i + 1) % full_attention_interval == 0
                    ? "full_attention"
                    : "linear_attention");
        }
        config_json["layer_types"] = std::move(layer_types);
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    // Family-owned checks for this construction graph only, before the first owner allocation.
    const auto positive = [&](const char *key) {
        if (model_config->get<int64_t>(key) <= 0)
            throw std::invalid_argument(std::string("Qwen3.5 requires positive ") + key);
        return model_config->get<size_t>(key);
    };
    const size_t limit = std::numeric_limits<size_t>::max();
    const auto product = [limit](size_t a, size_t b) {
        if (b && a > limit / b) throw std::length_error("Qwen3.5 construction size overflow");
        return a * b;
    };
    const size_t hidden = positive("hidden_size"), vocab = positive("vocab_size");
    const auto dtype = model_config->get_dtype();
    const size_t bytes = infinilm::runtime::dsize(dtype);
    (void)product(product(hidden, vocab), bytes);
    const double epsilon = model_config->get<double>("rms_norm_eps");
    if (!std::isfinite(epsilon) || epsilon <= 0) throw std::invalid_argument("Qwen3.5 requires finite positive rms_norm_eps");
    const auto layer_types = model_config->get<std::vector<std::string>>("layer_types");
    if (layer_types.size() != static_cast<size_t>(layers_signed))
        throw std::invalid_argument("Qwen3.5 layer_types must match num_hidden_layers");
    bool full = false, linear = false;
    for (const auto &type : layer_types) {
        if (type == "full_attention") full = true;
        else if (type == "linear_attention") linear = true;
        else throw std::invalid_argument("Qwen3.5 has unsupported layer_type");
    }
    if (model_config->get<std::string>("model_type") == "qwen3_5") {
        const size_t mid = positive("intermediate_size");
        (void)product(product(product(mid, 2), hidden), bytes);
    } else {
        const size_t mid = positive("moe_intermediate_size"), shared = positive("shared_expert_intermediate_size");
        const size_t experts = positive("num_experts"), topk = positive("num_experts_per_tok");
        if (topk > experts) throw std::invalid_argument("Qwen3.5 MoE topk exceeds expert count");
        (void)product(product(product(product(experts, mid), 2), hidden), bytes);
        (void)product(product(product(shared, 2), hidden), bytes);
    }
    if (linear) {
        const size_t key = product(positive("linear_num_key_heads"), positive("linear_key_head_dim"));
        const size_t value = product(positive("linear_num_value_heads"), positive("linear_value_head_dim"));
        const auto kernel = model_config->get<int64_t>("linear_conv_kernel_dim");
        if (kernel < 0) throw std::invalid_argument("Qwen3.5 requires nonnegative linear_conv_kernel_dim");
        const size_t twice_key = product(key, 2);
        if (twice_key > limit - value) throw std::length_error("Qwen3.5 linear projection size overflow");
        (void)product(product(twice_key + value, hidden), bytes);
        (void)product(product(twice_key + value, static_cast<size_t>(kernel)), bytes);
        // Legal kernel0/1 remain zero-length history; no cache/state policy change.
    }
    if (full) {
        const size_t heads = positive("num_attention_heads"), kv = positive("num_key_value_heads"), dim = positive("head_dim");
        if (dim < 2 || heads % kv) throw std::invalid_argument("Qwen3.5 requires valid GQA/RoPE dimensions");
        (void)positive("max_position_embeddings");
        const auto &rope = config_json.at("rope_parameters");
        const double scaled = static_cast<double>(dim) * rope.at("partial_rotary_factor").get<double>();
        if (!std::isfinite(scaled) || scaled < 0 || scaled >= static_cast<double>(std::numeric_limits<long long>::max()))
            throw std::invalid_argument("Qwen3.5 rotary dimension is invalid");
        size_t rotary = std::clamp(static_cast<size_t>(std::llround(scaled)), static_cast<size_t>(2), dim);
        if (rotary % 2) --rotary;
        rotary = std::max(rotary, static_cast<size_t>(2)); // Preserve existing attention rounding/clamp.
        const auto sections = rope.at("mrope_section").get<std::vector<int>>();
        size_t sum = 0;
        if (sections.size() != 3) throw std::invalid_argument("Qwen3.5 mrope_section must have 3 elements");
        for (int section : sections) {
            if (section <= 0 || static_cast<size_t>(section) > limit - sum)
                throw std::invalid_argument("Qwen3.5 mrope_section must be positive");
            sum += static_cast<size_t>(section);
        }
        if (sum != rotary / 2) throw std::invalid_argument("MRoPE section sum must equal rotary_dim / 2");
        (void)rope.at("mrope_interleaved").get<bool>();
        const double theta = rope.at("rope_theta").get<double>();
        if (!std::isfinite(theta) || theta <= 0) throw std::invalid_argument("Qwen3.5 requires finite positive rope_theta");
    }
    if (config_json.contains("vision_config") && !config_json["vision_config"].is_null()) {
        const auto &vision = config_json.at("vision_config");
        if (!vision.is_object()) throw std::invalid_argument("Qwen3.5 vision_config must be an object");
        const auto vdim = [&](const char *key, int fallback) {
            const int64_t size = vision.value(key, static_cast<int64_t>(fallback));
            if (size <= 0 || size > std::numeric_limits<int>::max())
                throw std::invalid_argument(std::string("Qwen3.5 vision requires valid ") + key);
            return static_cast<size_t>(size); // Original value(key,int) representable domain.
        };
        const size_t vh = vdim("hidden_size", 1152), vn = vdim("num_heads", 16);
        const int64_t depth = vision.value("depth", vision.value("num_hidden_layers", int64_t{27}));
        if (depth < 0 || depth > std::numeric_limits<int>::max()) throw std::invalid_argument("Qwen3.5 vision depth is invalid");
        if (depth > 0 && (vh % vn || (vh / vn) % 4)) throw std::invalid_argument("Qwen3.5 vision requires divisible 2D RoPE geometry");
        const size_t vm = vdim("intermediate_size", 4304), merge = vdim("spatial_merge_size", 2);
        const size_t merged = product(product(vh, merge), merge);
        (void)product(product(merged, merged), bytes);
        (void)product(product(vh, vm), bytes);
        (void)product(product(vh, vdim("num_position_embeddings", 2304)), bytes);
        const auto patch_dim = [&](const char *key, size_t fallback) {
            if (!vision.contains(key) || vision.at(key).is_null()) return fallback;
            const auto &value = vision.at(key);
            const auto &scalar = value.is_array() && !value.empty() ? value.at(0) : value;
            if (value.is_array() && value.empty()) return fallback;
            if (scalar.get<int64_t>() <= 0) throw std::invalid_argument("Qwen3.5 vision patch dimensions must be positive");
            return scalar.get<size_t>();
        };
        const size_t patch = patch_dim("patch_size", 16), temporal = patch_dim("temporal_patch_size", 2);
        (void)product(product(product(product(product(vh, vdim("in_channels", 3)), temporal), patch), patch), bytes);
        (void)product(product(merged, vdim("out_hidden_size", static_cast<int>(vh))), bytes);
        const double norm_eps = vision.value("layer_norm_eps", vision.value("rms_norm_eps", 1e-6));
        if (!std::isfinite(norm_eps) || norm_eps <= 0) throw std::invalid_argument("Qwen3.5 vision norm epsilon is invalid");
    }
    return model_config;
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_5" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_5::create_qwen3_5_model_config: model_type is not qwen3_5");
    }
    return prepare_qwen3_5_model_config(model_config);
}

} // namespace infinilm::models::qwen3_5

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    qwen3_5,
    infinilm::models::qwen3_5::Qwen35ForCausalLM,
    infinilm::models::qwen3_5::create_qwen3_5_model_config);
} // namespace
