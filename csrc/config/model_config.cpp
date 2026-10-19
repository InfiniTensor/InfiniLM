#include "model_config.hpp"
#include "../layers/rotary_embedding/rope.hpp"

namespace infinilm::config {
ModelConfig::ModelConfig(const nlohmann::json &json) : config_json(json) {}

ModelConfig::ModelConfig(const std::string &path) {
    std::ifstream file(path);
    if (file.is_open()) {
        file >> config_json;
        file.close();
    } else {
        throw std::runtime_error("Could not open config file: " + path);
    }
}

std::shared_ptr<infinilm::layers::RoPE> ModelConfig::get_or_create_rope(
    const std::string &key,
    const std::function<std::shared_ptr<infinilm::layers::RoPE>()> &create) const {
    std::lock_guard<std::mutex> lock(rope_cache_->mutex);
    auto it = rope_cache_->entries.find(key);
    if (it != rope_cache_->entries.end()) {
        return it->second;
    }
    auto rope = create();
    rope_cache_->entries.emplace(key, rope);
    return rope;
}

infinilm::quantization::QuantScheme
ModelConfig::get_quant_scheme() const {
    return get_quant_config().get_quant_scheme();
}

infinilm::runtime::DataType ModelConfig::get_dtype() const {
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

size_t ModelConfig::get_rotary_dim() const {
    size_t head_dim = get_head_dim();
    double partial_rotary_factor = get_or<double>("partial_rotary_factor", 1.0);

    if (partial_rotary_factor <= 0.0 || partial_rotary_factor >= 1.0) {
        return head_dim;
    }

    size_t rotary_dim = static_cast<size_t>(std::llround(
        static_cast<double>(head_dim) * partial_rotary_factor));
    rotary_dim = std::clamp(rotary_dim, static_cast<size_t>(2), head_dim);

    if (rotary_dim % 2 != 0) {
        rotary_dim -= 1;
    }
    return std::max(rotary_dim, static_cast<size_t>(2));
}

std::ostream &operator<<(std::ostream &os, const ModelConfig &config) {
    os << config.config_json.dump(4);
    return os;
}

} // namespace infinilm::config
