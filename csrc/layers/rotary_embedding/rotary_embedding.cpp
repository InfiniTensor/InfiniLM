#include "rotary_embedding.hpp"
#include "../../config/model_config.hpp"
#include "rotary_embedding_factory.hpp"
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_map>

namespace infinilm::layers::rotary_embedding {

namespace {
// Cache dictionary to avoid redundant allocations of RoPE instances.
// thread_local ensures it is only visible within this compilation unit.
thread_local std::unordered_map<std::string, std::shared_ptr<infinicore::nn::RoPE>> _ROPE_DICT;

std::string make_cache_key(size_t head_dim,
                           size_t rotary_dim,
                           size_t max_position_embeddings,
                           double rope_theta,
                           infinicore::nn::RoPE::Algo algo,
                           const infinicore::DataType &dtype,
                           const infinicore::Device &device,
                           const std::shared_ptr<infinicore::nn::RopeScalingConfig> &scaling,
                           const std::optional<std::vector<int>> &mrope_section,
                           bool mrope_interleaved) {
    std::ostringstream oss;
    oss << "scaling_" << (scaling ? typeid(*scaling).name() : "none")
        << "_head_" << head_dim
        << "_rotary_" << rotary_dim
        << "_max_" << max_position_embeddings
        << "_theta_" << std::setprecision(17) << rope_theta
        << "_algo_" << static_cast<int>(algo)
        << "_dtype_" << static_cast<int>(dtype)
        << "_dev" << device.toString();

    if (mrope_section.has_value()) {
        oss << "_mrope";
        for (const auto section : mrope_section.value()) {
            oss << '_' << section;
        }
        oss << "_mrope_interleaved_" << (mrope_interleaved ? 1 : 0);
    }
    return oss.str();
}

} // namespace

std::shared_ptr<infinicore::nn::RoPE>
get_rope(size_t head_dim,
         size_t rotary_dim,
         size_t max_position_embeddings,
         double rope_theta,
         infinicore::nn::RoPE::Algo algo,
         const infinicore::DataType &dtype,
         const infinicore::Device &device,
         std::shared_ptr<infinicore::nn::RopeScalingConfig> scaling,
         std::optional<std::vector<int>> mrope_section,
         bool mrope_interleaved) {
    const auto cache_key = make_cache_key(head_dim,
                                          rotary_dim,
                                          max_position_embeddings,
                                          rope_theta,
                                          algo,
                                          dtype,
                                          device,
                                          scaling,
                                          mrope_section,
                                          mrope_interleaved);
    auto it = _ROPE_DICT.find(cache_key);
    if (it != _ROPE_DICT.end()) {
        return it->second;
    }

    auto rope = std::make_shared<infinicore::nn::RoPE>(head_dim,
                                                       rotary_dim,
                                                       max_position_embeddings,
                                                       rope_theta,
                                                       algo,
                                                       dtype,
                                                       device,
                                                       scaling,
                                                       mrope_section,
                                                       mrope_interleaved);
    _ROPE_DICT.emplace(cache_key, rope);
    return rope;
}

std::shared_ptr<infinicore::nn::RoPE>
get_rope(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
         const infinicore::Device &device) {

    size_t rotary_dim = model_config->get_rotary_dim();
    size_t head_dim = model_config->get_head_dim();
    auto scaling = make_scaling_config(model_config);

    const auto &dtype = model_config->get_dtype();
    size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    double rope_theta = model_config->get<double>("rope_theta");
    infinicore::nn::RoPE::Algo algo = model_config->get_rope_algo();

    return get_rope(head_dim,
                    rotary_dim,
                    max_position_embeddings,
                    rope_theta,
                    algo,
                    dtype,
                    device,
                    scaling,
                    std::nullopt,
                    false);
}

} // namespace infinilm::layers::rotary_embedding
