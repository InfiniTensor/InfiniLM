#include "cache_interface.hpp"
#include "dynamic_cache/dynamic_cache.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::cache {

std::shared_ptr<CacheInterface> CacheInterface::create(const CacheConfig &config) {
    switch (config.type) {
    case CacheType::DYNAMIC:
        return std::make_shared<DynamicCache>(config);

    case CacheType::PAGED:
        // Return PagedCache when implemented
        // return std::make_shared<PagedCache>(config);
        spdlog::warn("PagedCache not yet implemented, falling back to DynamicCache");
        return std::make_shared<DynamicCache>(config);

    default:
        spdlog::error("Unknown cache type: {}", static_cast<int>(config.type));
        throw std::runtime_error("Unknown cache type");
    }
}

} // namespace infinilm::cache
