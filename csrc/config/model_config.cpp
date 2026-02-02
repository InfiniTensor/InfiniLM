#include "model_config.hpp"

namespace infinilm::config {
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

infinicore::quantization::QuantScheme
ModelConfig::get_quant_scheme() const {
    if (quant_config.get_quant_scheme() != infinicore::quantization::QuantScheme::NONE) {
        return quant_config.get_quant_scheme();
    } else {
        return infinicore::quantization::QuantScheme::NONE;
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

    if (!rope_scaling.contains("type")) {
        throw std::runtime_error("rope_scaling must contain 'type' field");
    }

    std::string type_str = rope_scaling["type"].get<std::string>();
    if (type_str == "longrope") {
        // Required fields for LongRopeConfig
        if (!rope_scaling.contains("short_factor") || !rope_scaling.contains("long_factor") || !rope_scaling.contains("original_max_position_embeddings")) {
            throw std::runtime_error(
                "LongRopeConfig requires 'short_factor', 'long_factor', and 'original_max_position_embeddings'");
        }

        auto short_factor = rope_scaling["short_factor"].get<std::vector<float>>();
        auto long_factor = rope_scaling["long_factor"].get<std::vector<float>>();
        size_t original_max_position_embeddings = rope_scaling["original_max_position_embeddings"].get<size_t>();

        float factor = 1.0f;
        if (rope_scaling.contains("factor")) {
            factor = rope_scaling["factor"].get<float>();
        }

        return std::make_shared<infinicore::nn::RoPE::LongRopeConfig>(
            std::move(short_factor),
            std::move(long_factor),
            original_max_position_embeddings,
            factor);
    } else if (type_str == "default" || type_str == "none") {
        // Default scaling, no scaling applied
        return nullptr;
    } else {
        throw std::runtime_error("Unsupported rope_scaling type: " + type_str);
    }
}

infinicore::DataType
ModelConfig::get_dtype() const {
    try {
        std::string dtype_str = this->get<std::string>("torch_dtype");
        if (dtype_str == "float32") {
            return infinicore::DataType::F32;
        } else if (dtype_str == "float16") {
            return infinicore::DataType::F16;
        } else if (dtype_str == "bfloat16") {
            return infinicore::DataType::BF16;
        } else if (dtype_str == "int8") {
            return infinicore::DataType::I8;
        } else {
            throw std::runtime_error("Unsupported dtype string: " + dtype_str);
        }
    } catch (const std::exception &e) {
        throw std::runtime_error("Error getting dtype from config: " + std::string(e.what()));
    }
}
} // namespace infinilm::config
