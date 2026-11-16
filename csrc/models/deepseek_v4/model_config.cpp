#include "model_config.hpp"

#include "models/models_registry.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace infinilm::models::deepseek_v4 {

std::shared_ptr<config::ModelConfig> create_deepseek_v4_model_config(
    std::shared_ptr<config::ModelConfig> source) {
    if (!source || source->get_or<std::string>("model_type", "") != "deepseek_v4") {
        throw std::invalid_argument("DeepSeek V4 config requires model_type=deepseek_v4");
    }

    auto json = source->get_config_json();
    auto normalized = std::make_shared<config::ModelConfig>(json);
    const auto head = normalized->get<int64_t>("head_dim");
    const auto rotary = normalized->get<int64_t>("qk_rope_head_dim");
    const auto layers = normalized->get<int64_t>("num_hidden_layers");
    if (head <= 0 || rotary <= 0 || rotary > head || rotary % 2 || layers <= 0) {
        throw std::invalid_argument("Invalid DeepSeek V4 head or layer geometry");
    }
    const auto ratios = normalized->get<std::vector<int64_t>>("compress_ratios");
    if (ratios.size() < static_cast<size_t>(layers)
        || !std::all_of(ratios.begin(), ratios.end(), [](auto x) {
            return x == 0 || x == 4 || x == 128;
        })) {
        throw std::invalid_argument("Invalid DeepSeek V4 compression ratios");
    }
    if (json.contains("qk_nope_head_dim")
        && normalized->get<int64_t>("qk_nope_head_dim") != head - rotary) {
        throw std::invalid_argument("DeepSeek V4 rotary and non-rotary dimensions disagree");
    }
    auto &out = source->get_config_json();
    out["qk_nope_head_dim"] = head - rotary;
    if ((!out.contains("quantization_config") || out["quantization_config"].is_null())
        && out.contains("compression_config")) {
        out["quantization_config"] = out["compression_config"];
    }
    return source;
}

} // namespace infinilm::models::deepseek_v4
