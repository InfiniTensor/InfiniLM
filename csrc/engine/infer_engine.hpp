#pragma once

#include "../cache/cache_manager.hpp"
#include "distributed/distributed.hpp"
#include "infinicore/tensor.hpp"
#include "rank_worker.hpp"

#include <any>
#include <vector>

namespace infinilm::engine {

class InferEngine {
public:
    InferEngine(
        const std::any &config,
        const distributed::DistConfig &distributed_config = distributed::DistConfig(),
        infinicore::Device::Type device_type = infinicore::context::getDevice().getType(),
        cache::CacheType cache_type = cache::CacheType::DYNAMIC);

    // Load a parameter to all workers (each can extract its shard inside RankWorker)
    void load_param(const std::string &name, const infinicore::Tensor &param);

    // return the parameters (i.e. weights and biases).
    std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> state_dict();

    // Run a single forward pass on all workers and return the outputs from all ranks
    infinicore::Tensor generate(const infinicore::Tensor &input_ids,
                                const infinicore::Tensor &position_ids);

    // Reset the internal cache in all workers (clears state between generations)
    // By default, this is synchronous (blocks until reset completes).
    // If async=true, this becomes asynchronous (unstable - use with caution).
    void reset_cache(size_t pos = 0, bool async = false);

    ~InferEngine();

    const distributed::DistConfig &get_dist_config() const;

    // Get cache manager for external control
    cache::CacheManager &get_cache_manager() { return *cache_manager_; }
    const cache::CacheManager &get_cache_manager() const { return *cache_manager_; }

protected:
    std::vector<std::unique_ptr<RankWorker>> workers_;
    distributed::CommunicationGroup communication_group_;
    std::any model_config_;
    std::unique_ptr<cache::CacheManager> cache_manager_;
};

} // namespace infinilm::engine
