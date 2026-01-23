#pragma once

#include "../config/global_config.hpp"
#include "../models/infinilm_model.hpp"
#include "../models/llama/llama_config.hpp"
#include "distributed/distributed.hpp"
#include "infinicore/tensor.hpp"
#include "rank_worker.hpp"

#include <optional>
#include <vector>

namespace infinilm::engine {

class InferEngine {
public:
    using Input = RankWorker::Input;

    using Output = RankWorker::Output;

    // Updated constructor: accept CacheConfig instead of CacheType
    InferEngine(
        const distributed::DistConfig &distributed_config = distributed::DistConfig(),
        infinicore::Device::Type device_type = infinicore::context::getDevice().getType(),
        const cache::CacheConfig *cache_config = nullptr,
        const std::string &modle_path = "");

    // Load a parameter to all workers (each can extract its shard inside RankWorker)
    void load_param(const std::string &name, const infinicore::Tensor &param);

    // return the parameters (i.e. weights and biases).
    std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> state_dict();

    // Run a single forward pass on all workers and return the outputs from all ranks
    Output forward(const Input &input);

    void reset_cache(const cache::CacheConfig *new_config);

    ~InferEngine();

    const distributed::DistConfig &get_dist_config() const;

    // Get current KV configuration
    const cache::CacheConfig *get_cache_config() const { return cache_config_.get(); }

protected:
    std::vector<std::unique_ptr<RankWorker>> workers_;
    distributed::CommunicationGroup communication_group_;
    // const InfinilmModel::Config &model_config_;
    std::unique_ptr<cache::CacheConfig> cache_config_;
    std::shared_ptr<infinilm::config::global_config::GlobalConfig> global_config_;
};

} // namespace infinilm::engine
