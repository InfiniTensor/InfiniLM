#include "../../config/model_config.hpp"
#include "infinicore/nn/rope_scaling_configs.hpp"
#include "rotary_embedding_factory.hpp"
#include <vector>

namespace infinilm::layers::rotary_embedding {
namespace {
/**
 * @brief Default creator for types that apply no scaling.
 * Returns nullptr, which the InfiniCore RoPE layer interprets as a 1.0x pass-through.
 */
std::shared_ptr<infinicore::nn::RopeScalingConfig>
create_default_scaling_config(const std::shared_ptr<config::ModelConfig> &) {
    return nullptr;
}

// TODO(rubik) create_dynamic_scaling

/**
 * @brief Creator function for LongRoPE scaling configuration.
 * Extracts 'short_factor', 'long_factor', etc., from the model config.
 */
std::shared_ptr<infinicore::nn::RopeScalingConfig>
create_longrope_config(const std::shared_ptr<config::ModelConfig> &cfg) {
    const auto &rope_scaling = cfg->get_config_json()["rope_scaling"];

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

    return std::make_shared<infinicore::nn::LongRopeScalingConfig>(
        std::move(short_factor),
        std::move(long_factor),
        original_max_position_embeddings,
        factor);
}

/**
 * @brief Creator function for Llama3 RoPE scaling configuration.
 * Extracts 'factor', 'low_freq_factor', 'high_freq_factor', and
 * 'original_max_position_embeddings' from the model config.
 */
std::shared_ptr<infinicore::nn::RopeScalingConfig>
create_llama3_scaling_config(const std::shared_ptr<config::ModelConfig> &cfg) {
    const auto &rope_scaling = cfg->get_config_json()["rope_scaling"];

    // Validate required fields for Llama3 scaling
    if (!rope_scaling.contains("factor") || !rope_scaling.contains("low_freq_factor") || !rope_scaling.contains("high_freq_factor") || !rope_scaling.contains("original_max_position_embeddings")) {
        throw std::runtime_error(
            "Llama3RopeScalingConfig requires 'factor', 'low_freq_factor', 'high_freq_factor', and 'original_max_position_embeddings'");
    }

    float factor = rope_scaling["factor"].get<float>();
    float low_freq_factor = rope_scaling["low_freq_factor"].get<float>();
    float high_freq_factor = rope_scaling["high_freq_factor"].get<float>();
    size_t original_max_position_embeddings = rope_scaling["original_max_position_embeddings"].get<size_t>();

    return std::make_shared<infinicore::nn::Llama3RopeScalingConfig>(
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings);
}

/**
 * @brief Creator function for YaRN RoPE scaling configuration.
 */
std::shared_ptr<infinicore::nn::RopeScalingConfig>
create_yarn_scaling_config(const std::shared_ptr<config::ModelConfig> &cfg) {
    const auto &rope_scaling = cfg->get_config_json()["rope_scaling"];

    if (!rope_scaling.contains("factor") || !rope_scaling.contains("original_max_position_embeddings")) {
        throw std::runtime_error(
            "YarnRopeScalingConfig requires 'factor' and 'original_max_position_embeddings'");
    }

    const float factor = rope_scaling["factor"].get<float>();
    const size_t original_max_position_embeddings = rope_scaling["original_max_position_embeddings"].get<size_t>();
    const int beta_fast = static_cast<int>(rope_scaling.value("beta_fast", 32.0f));
    const int beta_slow = static_cast<int>(rope_scaling.value("beta_slow", 1.0f));
    const float mscale = rope_scaling.value("mscale", 1.0f);
    const float mscale_all_dim = rope_scaling.value("mscale_all_dim", 0.0f);
    const size_t rotary_dim = cfg->get_rotary_dim();
    const float rope_theta = static_cast<float>(cfg->get<double>("rope_theta"));

    return std::make_shared<infinicore::nn::YarnRopeScalingConfig>(
        factor, original_max_position_embeddings, rotary_dim, rope_theta,
        beta_fast, beta_slow, mscale, mscale_all_dim);
}

} // anonymous namespace

// Static self-registration block
// Registers creator functions into the factory registry upon program startup.
static bool _registered = []() {
    auto &registry = get_scaling_registry();
    registry["default"] = create_default_scaling_config;
    registry["none"] = create_default_scaling_config;
    registry["dynamic"] = create_default_scaling_config;
    registry["longrope"] = create_longrope_config;
    registry["llama3"] = create_llama3_scaling_config;
    registry["yarn"] = create_yarn_scaling_config;
    return true;
}();

} // namespace infinilm::layers::rotary_embedding
