#include "rotary_embedding.hpp"
#include "../../config/model_config.hpp"
#include "rotary_embedding_factory.hpp"
#include <memory>
#include <string>

namespace infinilm::layers::rotary_embedding {

// Cache dictionary to avoid redundant allocations of RoPE instances.
// thread_local ensures it is only visible within this compilation unit.
thread_local std::unordered_map<std::string, std::shared_ptr<infinicore::nn::RoPE>> _ROPE_DICT;

std::shared_ptr<infinicore::nn::RoPE>
get_rope(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
         const infinicore::Device &device) {

    // 1. Compute the actual rotary dimension
    size_t rotary_dim = model_config->get_rotary_dim();
    size_t head_dim = model_config->get_head_dim();

    // 2. Resolve scaling config via the internal factory
    auto scaling = make_scaling_config(model_config);

    // 3. Cache key must include rotary_dim AND the actual scaling type
    //    to avoid reusing the same RoPE instance across models with different settings
    //    (Enhancement: dynamically determine scaling_type instead of hardcoding "default")
    std::string scaling_type_str = "default";
    if (scaling) {
        // Assuming we can get the type string from the JSON for cache key generation,
        // or ideally, ScalingConfig should have a virtual std::string type_name() const method.
        // Here we read it from JSON for the cache key purpose only, keeping it decoupled from InfiniCore.
        const auto &rope_scaling_json = model_config->get_config_json()["rope_scaling"];
        if (rope_scaling_json.contains("type")) {
            scaling_type_str = rope_scaling_json["type"].get<std::string>();
        } else if (rope_scaling_json.contains("rope_type")) {
            scaling_type_str = rope_scaling_json["rope_type"].get<std::string>();
        }
    }

    std::string cache_key = scaling_type_str + "_rope_dim_" + std::to_string(rotary_dim)
                          + "_dev_" + device.toString();
    auto it = _ROPE_DICT.find(cache_key);
    if (it != _ROPE_DICT.end()) {
        return it->second;
    }

    const auto &dtype = model_config->get_dtype();
    size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    double rope_theta = model_config->get<double>("rope_theta");

    infinicore::nn::RoPE::Algo algo = model_config->get_rope_algo();
    auto rope = std::make_shared<infinicore::nn::RoPE>(head_dim, rotary_dim, max_position_embeddings, rope_theta,
                                                       algo, dtype, device, scaling);

    _ROPE_DICT.emplace(cache_key, rope);
    return rope;
}

} // namespace infinilm::layers::rotary_embedding
