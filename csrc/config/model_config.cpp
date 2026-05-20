#include "model_config.hpp"

namespace infinilm::config {
ModelConfig::ModelConfig(const nlohmann::json &json) : config_json(json) {
    this->quant_config = QuantConfig(config_json["quantization_config"]);
};

ModelConfig::ModelConfig(const std::string &path) {
    std::ifstream file(path);
    if (file.is_open()) {
        file >> config_json;
        file.close();
    } else {
        throw std::runtime_error("Could not open config file: " + path);
    }
    this->quant_config = QuantConfig(config_json["quantization_config"]);
}

infinilm::quantization::QuantScheme
ModelConfig::get_quant_scheme() const {
    if (quant_config.get_quant_scheme() != infinilm::quantization::QuantScheme::NONE) {
        return quant_config.get_quant_scheme();
    } else {
        return infinilm::quantization::QuantScheme::NONE;
    }
}

std::shared_ptr<infinicore::nn::RoPE::ScalingConfig>
ModelConfig::get_rope_scaling() const {
    if (!config_json.contains("rope_scaling") || config_json["rope_scaling"].is_null()) {
        return nullptr;
    }

    const auto &rope_scaling = config_json["rope_scaling"];
    if (!rope_scaling.is_object()) {
        throw std::runtime_error("rope_scaling must be an object");
    }

    std::string type_str;
    if (rope_scaling.contains("type")) {
        type_str = rope_scaling["type"].get<std::string>();
    } else if (rope_scaling.contains("rope_type")) {
        type_str = rope_scaling["rope_type"].get<std::string>();
    } else {
        throw std::runtime_error("rope_scaling must contain 'type' or 'rope_type' field");
    }

    // SGLang's style if-else routing in python/srt/layers/rotary_embedding.py:get_rope
    if (type_str == "llama3") {
        return createLlama3Scaling(rope_scaling);
    } else if (type_str == "longrope") {
        return createLongRopeScaling(rope_scaling);
    } else if (type_str == "default") {
        return createDefaultScaling(rope_scaling);
    } else if (type_str == "none") {
        return createNoneScaling(rope_scaling);
    } else if (type_str == "dynamic") {
        return createDynamicScaling(rope_scaling);
    }

    throw std::runtime_error("Unsupported rope_scaling type: " + type_str);
}

std::shared_ptr<infinicore::nn::RoPE::ScalingConfig>
ModelConfig::createDefaultScaling(const nlohmann::json &rope_scaling) const {
    return nullptr;
}

std::shared_ptr<infinicore::nn::RoPE::ScalingConfig>
ModelConfig::createNoneScaling(const nlohmann::json &rope_scaling) const {
    return nullptr;
}

std::shared_ptr<infinicore::nn::RoPE::ScalingConfig>
ModelConfig::createDynamicScaling(const nlohmann::json &rope_scaling) const {
    // [TODO]Dynamic scaling: currently not handling extended sequence lengths
    return nullptr;
}

std::shared_ptr<infinicore::nn::RoPE::ScalingConfig>
ModelConfig::createLlama3Scaling(const nlohmann::json &rope_scaling) const {
    // Native support for Llama 3.1 frequency-aware RoPE scaling
    // equivalent to SGLang/HuggingFace implementations

    // 1. Validate and extract Llama3 specific parameters
    const std::vector<std::string> required_keys = {
        "factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"};
    for (const auto &key : required_keys) {
        if (!rope_scaling.contains(key)) {
            throw std::runtime_error("Llama3RoPE requires '" + key + "' in rope_scaling");
        }
    }

    const double factor = rope_scaling["factor"].get<double>();
    const double low_freq_factor = rope_scaling["low_freq_factor"].get<double>();
    const double high_freq_factor = rope_scaling["high_freq_factor"].get<double>();
    const size_t orig_max_pos = rope_scaling["original_max_position_embeddings"].get<size_t>();

    // 2. Validate and extract model base parameters
    if (!config_json.contains("hidden_size") || !config_json.contains("num_attention_heads")) {
        throw std::runtime_error("Llama3RoPE requires 'hidden_size' and 'num_attention_heads' in config");
    }
    const size_t hidden_size = config_json["hidden_size"].get<size_t>();
    const size_t num_heads = config_json["num_attention_heads"].get<size_t>();
    const size_t head_dim = hidden_size / num_heads;
    const double theta = config_json.value("rope_theta", 10000.0);

    // 3. Pre-compute smooth factors based on wavelength
    constexpr double kPi = 3.14159265358979323846;
    const size_t cache_dim = head_dim / 2;
    const double low_freq_wavelen = static_cast<double>(orig_max_pos) / low_freq_factor;
    const double high_freq_wavelen = static_cast<double>(orig_max_pos) / high_freq_factor;

    const bool has_smooth_range = (high_freq_factor != low_freq_factor);
    const double smooth_denom = has_smooth_range ? (high_freq_factor - low_freq_factor) : 1.0;

    std::vector<float> smooth_factors(cache_dim);
    for (size_t j = 0; j < cache_dim; ++j) {
        const double exponent = 2.0 * static_cast<double>(j) / static_cast<double>(head_dim);
        const double inv_freq = 1.0 / std::pow(theta, exponent);
        const double wavelen = 2.0 * kPi / inv_freq;

        if (wavelen < high_freq_wavelen) {
            // High frequency: no scaling (freq_scale = 1.0)
            smooth_factors[j] = 1.0f;
        } else if (wavelen > low_freq_wavelen) {
            // Low frequency: full scaling (freq_scale = 1.0 / factor)
            smooth_factors[j] = static_cast<float>(factor);
        } else {
            // Mid frequency: smooth frequency interpolation
            //
            // Equivalent to SGLang's implementation:
            //   smooth = (orig_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            //   new_freq = (1 - smooth) * inv_freq / factor + smooth * inv_freq
            //   => freq_scale = (1 - smooth) / factor + smooth
            //
            // Since LongRopeConfig applies factors as wavelength multipliers
            // (i.e., new_freq = inv_freq / scale), the required smooth_factor
            // is the inverse of the frequency scale.
            //
            double smooth = 0.0;
            if (has_smooth_range) {
                smooth = (static_cast<double>(orig_max_pos) / wavelen - low_freq_factor) / smooth_denom;
            }
            const double freq_scale = (1.0 - smooth) / factor + smooth;
            smooth_factors[j] = static_cast<float>(1.0 / freq_scale);
        }
    }

    // 4. Adapt to LongRopeConfig
    // - short_factor and long_factor use the same smooth_factors
    // - Pass factor=1.0f to bypass the amplitude scaling sqrt(log(...)) in LongRopeConfig constructor
    return std::make_shared<infinicore::nn::RoPE::LongRopeConfig>(
        smooth_factors, // short_factor
        smooth_factors, // long_factor
        orig_max_pos,
        1.0f // Force 1.0f to disable amplitude scaling
    );
}

std::shared_ptr<infinicore::nn::RoPE::ScalingConfig>
ModelConfig::createLongRopeScaling(const nlohmann::json &rope_scaling) const {
    const std::vector<std::string> required_keys = {"short_factor", "long_factor", "original_max_position_embeddings"};
    for (const auto &key : required_keys) {
        if (!rope_scaling.contains(key)) {
            throw std::runtime_error("LongRopeConfig requires '" + key + "' in rope_scaling");
        }
    }

    auto short_factor = rope_scaling["short_factor"].get<std::vector<float>>();
    auto long_factor = rope_scaling["long_factor"].get<std::vector<float>>();
    const size_t orig_max_pos = rope_scaling["original_max_position_embeddings"].get<size_t>();
    const float factor = rope_scaling.value("factor", 1.0f);

    return std::make_shared<infinicore::nn::RoPE::LongRopeConfig>(
        std::move(short_factor),
        std::move(long_factor),
        orig_max_pos,
        factor);
}

infinicore::DataType ModelConfig::get_dtype() const {
    std::string dtype_str;
    if (config_json.contains("dtype")) {
        dtype_str = this->get<std::string>("dtype");
    } else if (config_json.contains("torch_dtype")) {
        dtype_str = this->get<std::string>("torch_dtype");
    } else {
        throw std::runtime_error("ModelConfig::get_dtype(): No dtype or torch_dtype found in config");
    }

    return parse_dtype(dtype_str);
}

std::ostream &operator<<(std::ostream &os, const ModelConfig &config) {
    os << config.config_json.dump(4);
    return os;
}

} // namespace infinilm::config
