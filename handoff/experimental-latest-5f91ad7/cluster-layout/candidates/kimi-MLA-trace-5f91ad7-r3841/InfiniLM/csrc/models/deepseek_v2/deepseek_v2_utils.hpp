#pragma once

#include "../../config/model_config.hpp"

#include <cmath>
#include <memory>

namespace infinilm::models::deepseek_v2 {

inline float deepseek_v2_attention_softmax_scale(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
    float q_head_dim) {
    auto yarn_get_mscale = [](float scale, float mscale) {
        if (scale <= 1.0f) {
            return 1.0f;
        }
        return 0.1f * mscale * std::log(scale) + 1.0f;
    };

    float scale = 1.0f / std::sqrt(q_head_dim);
    const auto &config_json = model_config->get_config_json();
    if (!config_json.contains("rope_scaling") || !config_json["rope_scaling"].is_object()) {
        return scale;
    }

    const auto &rope_scaling = config_json["rope_scaling"];
    const float mscale_all_dim = rope_scaling.value("mscale_all_dim", 0.0f);
    if (mscale_all_dim == 0.0f) {
        return scale;
    }

    const float scaling_factor = rope_scaling.value("factor", 1.0f);
    const float mscale = yarn_get_mscale(scaling_factor, mscale_all_dim);
    return scale * mscale * mscale;
}

} // namespace infinilm::models::deepseek_v2
