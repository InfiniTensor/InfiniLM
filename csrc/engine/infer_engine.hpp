#pragma once

#include "distributed/distributed.hpp"
#include "infinicore/tensor.hpp"
#include "rank_worker.hpp"

#include <any>
#include <vector>

namespace infinilm::engine {

class InferEngine {
public:
    // Updated constructor: accept CacheConfig instead of CacheType
    InferEngine(
        const std::any &config,
        const distributed::DistConfig &distributed_config = distributed::DistConfig(),
        infinicore::Device::Type device_type = infinicore::context::getDevice().getType(),
        const cache::CacheConfig &cache_config = cache::CacheConfig());

    // Load a parameter to all workers (each can extract its shard inside RankWorker)
    void load_param(const std::string &name, const infinicore::Tensor &param);

    // return the parameters (i.e. weights and biases).
    std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> state_dict();

    // Run a single forward pass on all workers and return the outputs from all ranks
    infinicore::Tensor generate(const infinicore::Tensor &input_ids,
                                const infinicore::Tensor &position_ids);

    // Reset the internal cache pos in all workers (clears state between generations)
    void reset_cache(size_t pos = 0);

    // Overload: reset cache with new KV configuration
    void reset_cache(const cache::CacheConfig &new_config, size_t pos = 0);

    ~InferEngine();

    const distributed::DistConfig &get_dist_config() const;

    // Get current KV configuration
    const cache::CacheConfig &get_cache_config() const { return cache_config_; }

protected:
    std::vector<std::unique_ptr<RankWorker>> workers_;
    distributed::CommunicationGroup communication_group_;
    std::any model_config_;
    cache::CacheConfig cache_config_;
};

} // namespace infinilm::engine
