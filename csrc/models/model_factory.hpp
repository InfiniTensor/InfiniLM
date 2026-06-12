#pragma once

#include "infinilm_model.hpp"

namespace infinilm {

class InfinilmModelFactory {
public:
    static std::shared_ptr<InfinilmModel> createModel(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device,
        const cache::CacheConfig *cache = nullptr);
};
} // namespace infinilm
