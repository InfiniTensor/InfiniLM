#include "gemma4_config.hpp"

#include <cmath>
#include <stdexcept>

namespace infinilm::models::gemma4 {
namespace {

size_t read_positive_or_default(const nlohmann::json &json, const char *key, size_t fallback) {
    if (!json.contains(key) || json.at(key).is_null()) {
        return fallback;
    }
    const auto val = json.at(key).get<int64_t>();
    if (val <= 0) {
        return fallback;
    }
    return static_cast<size_t>(val);
}

size_t read_nonneg_or_default(const nlohmann::json &json, const char *key, size_t fallback) {
    if (!json.contains(key) || json.at(key).is_null()) {
        return fallback;
    }
    const auto val = json.at(key).get<int64_t>();
    if (val < 0) {
        throw std::invalid_argument(std::string("Gemma4 config field must be nonnegative: ") + key);
    }
    return static_cast<size_t>(val);
}

void parse_rope_section(const nlohmann::json &rope_params, const char *section_name,
                        double &theta, double &partial_factor, bool &is_proportional) {
    if (!rope_params.is_object() || !rope_params.contains(section_name)
        || !rope_params.at(section_name).is_object()) {
        return;
    }
    const auto &sec = rope_params.at(section_name);
    if (sec.contains("rope_theta") && !sec.at("rope_theta").is_null()) {
        theta = sec.at("rope_theta").get<double>();
    }
    if (sec.contains("partial_rotary_factor") && !sec.at("partial_rotary_factor").is_null()) {
        partial_factor = sec.at("partial_rotary_factor").get<double>();
    }
    if (sec.contains("rope_type") && sec.at("rope_type").is_string()) {
        is_proportional = (sec.at("rope_type").get<std::string>() == "proportional");
    }
}

} // namespace

Gemma4TextSpec parse_gemma4_text_spec(const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    if (!model_config) {
        throw std::invalid_argument("Gemma4 requires a non-null ModelConfig");
    }
    const auto &cfg = model_config->get_config_json();
    Gemma4TextSpec spec;
    spec.vocab_size = model_config->get<size_t>("vocab_size");
    spec.hidden_size = model_config->get<size_t>("hidden_size");
    spec.num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    spec.num_attention_heads = model_config->get<size_t>("num_attention_heads");
    spec.num_key_value_heads = model_config->get<size_t>("num_key_value_heads");
    spec.head_dim = read_positive_or_default(cfg, "head_dim", spec.hidden_size / spec.num_attention_heads);
    spec.global_head_dim = read_positive_or_default(cfg, "global_head_dim", spec.head_dim);
    spec.num_global_key_value_heads = read_positive_or_default(
        cfg, "num_global_key_value_heads", spec.num_key_value_heads);
    spec.intermediate_size = model_config->get<size_t>("intermediate_size");
    spec.max_position_embeddings = model_config->get_or<size_t>("max_position_embeddings", 131072);
    spec.sliding_window = read_positive_or_default(cfg, "sliding_window", 512);
    spec.num_kv_shared_layers = read_nonneg_or_default(cfg, "num_kv_shared_layers", 0);
    if (spec.num_kv_shared_layers > spec.num_hidden_layers) {
        throw std::invalid_argument("Gemma4 num_kv_shared_layers exceeds num_hidden_layers");
    }
    spec.first_kv_shared_layer_idx = spec.num_hidden_layers - spec.num_kv_shared_layers;
    spec.use_double_wide_mlp = model_config->get_or<bool>("use_double_wide_mlp", false);
    spec.attention_k_eq_v = model_config->get_or<bool>("attention_k_eq_v", false);
    spec.attention_bias = model_config->get_or<bool>("attention_bias", false);
    if (cfg.contains("use_bidirectional_attention") && cfg.at("use_bidirectional_attention").is_string()) {
        const auto mode = cfg.at("use_bidirectional_attention").get<std::string>();
        spec.use_bidirectional_vision_attention = (mode == "vision" || mode == "all");
    } else if (cfg.contains("use_bidirectional_attention") && cfg.at("use_bidirectional_attention").is_boolean()) {
        spec.use_bidirectional_vision_attention = cfg.at("use_bidirectional_attention").get<bool>();
    }
    spec.rms_norm_eps = model_config->get_or<double>("rms_norm_eps", 1e-6);
    if (cfg.contains("final_logit_softcapping") && !cfg.at("final_logit_softcapping").is_null()) {
        spec.final_logit_softcapping = cfg.at("final_logit_softcapping").get<double>();
    } else {
        spec.final_logit_softcapping = 0.0;
    }
    spec.hidden_size_per_layer_input = read_nonneg_or_default(cfg, "hidden_size_per_layer_input", 0);
    spec.vocab_size_per_layer_input = read_positive_or_default(cfg, "vocab_size_per_layer_input", spec.vocab_size);
    spec.pad_token_id = model_config->get_or<int64_t>("pad_token_id", 0);

    spec.enable_moe_block = model_config->get_or<bool>("enable_moe_block", false);
    if (spec.enable_moe_block) {
        spec.num_experts = model_config->get<size_t>("num_experts");
        spec.top_k_experts = model_config->get<size_t>("top_k_experts");
        spec.moe_intermediate_size = model_config->get<size_t>("moe_intermediate_size");
        if (spec.num_experts == 0 || spec.top_k_experts == 0 || spec.top_k_experts > spec.num_experts) {
            throw std::invalid_argument("Gemma4 MoE expert/top_k geometry is invalid");
        }
    }

    if (cfg.contains("rope_parameters") && cfg.at("rope_parameters").is_object()) {
        const auto &rp = cfg.at("rope_parameters");
        parse_rope_section(rp, "sliding_attention",
                           spec.sliding_rope_theta, spec.sliding_partial_rotary_factor, spec.sliding_is_proportional);
        parse_rope_section(rp, "full_attention",
                           spec.full_rope_theta, spec.full_partial_rotary_factor, spec.full_is_proportional);
    }

    std::vector<std::string> layer_types;
    if (cfg.contains("layer_types") && cfg.at("layer_types").is_array()) {
        layer_types = cfg.at("layer_types").get<std::vector<std::string>>();
    } else {
        layer_types.assign(spec.num_hidden_layers, "sliding_attention");
        if (spec.num_hidden_layers > 0) {
            layer_types.back() = "full_attention";
        }
    }
    if (layer_types.size() != spec.num_hidden_layers) {
        throw std::invalid_argument("Gemma4 layer_types length does not match num_hidden_layers");
    }

    // Track last non-shared layer index for each layer_type.
    size_t last_sliding_non_shared = spec.num_hidden_layers;
    size_t last_full_non_shared = spec.num_hidden_layers;
    spec.layers.reserve(spec.num_hidden_layers);
    for (size_t i = 0; i < spec.num_hidden_layers; ++i) {
        Gemma4LayerSpec ls;
        ls.layer_idx = i;
        ls.layer_type = layer_types[i];
        if (ls.layer_type == "sliding_attention") {
            ls.is_sliding = true;
            ls.head_dim = spec.head_dim;
            ls.num_kv_heads = spec.num_key_value_heads;
            ls.rope_theta = spec.sliding_rope_theta;
            ls.partial_rotary_factor = spec.sliding_partial_rotary_factor;
            ls.is_proportional_rope = spec.sliding_is_proportional;
        } else if (ls.layer_type == "full_attention") {
            ls.is_sliding = false;
            ls.head_dim = spec.global_head_dim;
            ls.num_kv_heads = spec.num_global_key_value_heads;
            ls.rope_theta = spec.full_rope_theta;
            ls.partial_rotary_factor = spec.full_partial_rotary_factor;
            ls.is_proportional_rope = spec.full_is_proportional;
        } else {
            throw std::invalid_argument("Gemma4 unsupported layer_type: " + ls.layer_type);
        }

        ls.is_kv_shared = (spec.num_kv_shared_layers > 0 && i >= spec.first_kv_shared_layer_idx);
        if (!ls.is_kv_shared) {
            ls.kv_source_layer_idx = i;
            if (ls.is_sliding) {
                last_sliding_non_shared = i;
            } else {
                last_full_non_shared = i;
            }
        } else {
            const size_t src = ls.is_sliding ? last_sliding_non_shared : last_full_non_shared;
            if (src >= spec.num_hidden_layers) {
                throw std::invalid_argument("Gemma4 KV-shared layer has no preceding non-shared layer of matching type");
            }
            ls.kv_source_layer_idx = src;
        }
        ls.use_k_eq_v = spec.attention_k_eq_v && !ls.is_sliding;
        ls.mlp_intermediate_size = spec.intermediate_size * ((spec.use_double_wide_mlp && ls.is_kv_shared) ? 2 : 1);
        spec.layers.push_back(std::move(ls));
    }

    return spec;
}

std::shared_ptr<infinilm::config::ModelConfig> prepare_gemma4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (!model_config) {
        throw std::invalid_argument("Gemma4 requires configuration");
    }
    nlohmann::json &config_json = model_config->get_config_json();
    if (config_json.contains("text_config") && config_json["text_config"].is_object()) {
        const nlohmann::json &text_config = config_json["text_config"];
        for (auto it = text_config.begin(); it != text_config.end(); ++it) {
            if (!config_json.contains(it.key()) || config_json[it.key()].is_null()) {
                config_json[it.key()] = it.value();
            }
        }
    }
    if (!config_json.contains("dtype") || config_json["dtype"].is_null()) {
        if (config_json.contains("torch_dtype") && !config_json["torch_dtype"].is_null()) {
            config_json["dtype"] = config_json["torch_dtype"];
        } else {
            config_json["dtype"] = "bfloat16";
        }
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    if (!config_json.contains("tie_word_embeddings")) {
        config_json["tie_word_embeddings"] = true;
    }
    if (!config_json.contains("position_id_axes")) {
        config_json["position_id_axes"] = 1;
    }
    // Validate full spec eagerly so broken configs fail before allocating weights.
    (void)parse_gemma4_text_spec(model_config);
    return model_config;
}

std::shared_ptr<infinilm::config::ModelConfig> create_gemma4_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if (model_type != "gemma4") {
        throw std::runtime_error("infinilm::models::gemma4::create_gemma4_model_config: model_type is not gemma4");
    }
    return prepare_gemma4_model_config(std::move(model_config));
}

std::shared_ptr<infinilm::config::ModelConfig> create_gemma4_text_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if (model_type != "gemma4_text") {
        throw std::runtime_error("infinilm::models::gemma4::create_gemma4_text_model_config: model_type is not gemma4_text");
    }
    return prepare_gemma4_model_config(std::move(model_config));
}

} // namespace infinilm::models::gemma4
