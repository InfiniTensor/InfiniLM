#include "rope.hpp"
#include "rope_scaling_configs.hpp"
#include "rotary_embedding.hpp"
#include "../../config/model_config.hpp"
#include "rotary_embedding_factory.hpp"
#include <iomanip>
#include <cstdint>
#include <utility>
#include "runtime/runtime.hpp"
#include <memory>
#include <sstream>
#include <string>
#include <typeinfo>

namespace infinilm::layers::rotary_embedding {

namespace {
std::string make_cache_key(size_t head_dim,
                           size_t rotary_dim,
                           size_t max_position_embeddings,
                           double rope_theta,
                           infinilm::layers::RoPE::Algo algo,
                           const infinilm::runtime::DataType &dtype,
                           const infinilm::runtime::Device &device,
                           const std::shared_ptr<infinilm::layers::RopeScalingConfig> &scaling,
                           const std::optional<std::vector<int>> &mrope_section,
                           bool mrope_interleaved,
                           const infinilm::config::ModelConfig &model_config) {
    std::ostringstream oss;
    oss << "scaling_" << (scaling ? typeid(*scaling).name() : "none")
        << "_head_" << head_dim
        << "_rotary_" << rotary_dim
        << "_max_" << max_position_embeddings
        << "_theta_" << std::setprecision(17) << rope_theta
        << "_algo_" << static_cast<int>(algo)
        << "_dtype_" << static_cast<int>(dtype)
        << "_dev" << device.ToString();

    if (mrope_section.has_value()) {
        oss << "_mrope";
        for (const auto section : mrope_section.value()) {
            oss << '_' << section;
        }
        oss << "_mrope_interleaved_" << (mrope_interleaved ? 1 : 0);
    }
    const auto &config = model_config.get_config_json();
    if (config.contains("rope_scaling")) {
        oss << "_scaling_params_" << config.at("rope_scaling").dump();
    }
    if (config.contains("rope_parameters")) {
        oss << "_rope_params_" << config.at("rope_parameters").dump();
    }
    return oss.str();
}

} // namespace

std::shared_ptr<infinilm::layers::RoPE>
get_rope(infinilm::runtime::Runtime &runtime, const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
         size_t head_dim,
         size_t rotary_dim,
         size_t max_position_embeddings,
         double rope_theta,
         infinilm::layers::RoPE::Algo algo,
         const infinilm::runtime::DataType &dtype,
         const infinilm::runtime::Device &device,
         std::shared_ptr<infinilm::layers::RopeScalingConfig> scaling,
         std::optional<std::vector<int>> mrope_section,
         bool mrope_interleaved) {
    runtime.assertOwnerThread();
    if (runtime.device() != device) throw std::invalid_argument("get_rope Runtime and device must agree");
    if (!model_config) {
        throw std::invalid_argument("get_rope requires a model owner");
    }
    const auto cache_key = make_cache_key(head_dim,
                                          rotary_dim,
                                          max_position_embeddings,
                                          rope_theta,
                                          algo,
                                          dtype,
                                          device,
                                          scaling,
                                          mrope_section,
                                          mrope_interleaved,
                                          *model_config)
        + "_runtime_owner_" + std::to_string(reinterpret_cast<std::uintptr_t>(&runtime));
    return model_config->get_or_create_rope(cache_key, [&] {
        return std::make_shared<infinilm::layers::RoPE>(runtime, head_dim,
                                                       rotary_dim,
                                                       max_position_embeddings,
                                                       rope_theta,
                                                       algo,
                                                       dtype,
                                                       device,
                                                       scaling,
                                                       mrope_section,
                                                       mrope_interleaved);
    });
}

std::shared_ptr<infinilm::layers::RoPE>
get_rope(infinilm::runtime::Runtime &runtime, const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
         const infinilm::runtime::Device &device) {

    runtime.assertOwnerThread();
    if (runtime.device() != device) throw std::invalid_argument("get_rope Runtime and device must agree");
    size_t rotary_dim = model_config->get_rotary_dim();
    size_t head_dim = model_config->get_head_dim();
    auto scaling = make_scaling_config(model_config);

    const auto &dtype = model_config->get_dtype();
    size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    double rope_theta = model_config->get<double>("rope_theta");
    infinilm::layers::RoPE::Algo algo = model_config->get_rope_algo();

    return get_rope(runtime, model_config,
                    head_dim,
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
