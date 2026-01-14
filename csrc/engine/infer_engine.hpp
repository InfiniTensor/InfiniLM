#pragma once

#include "../config/model_config.hpp"
#include "../models/infinilm_model.hpp"
#include "../models/llama/llama_config.hpp"
#include "distributed/distributed.hpp"
#include "infinicore/tensor.hpp"
#include "rank_barrier.hpp"
#include "rank_worker.hpp"

#include <optional>
#include <vector>

namespace infinilm::engine {

class InferEngine {
public:
    using Input = RankWorker::Input;

    using Output = RankWorker::Output;

    // Updated constructor: accept CacheConfig instead of CacheType
    /**
     * @deprecated This function is deprecated and will be REMOVED in the next major release (v0.2.0).
     *
     * ⚠️ DEVELOPMENT POLICY:
     *   - NO new development or feature additions permitted on this interface
     *   - Only critical bug fixes (security/stability) allowed until removal
     *   - All new code MUST migrate to the polymorphic overload below
     *
     * Replacement: Use the polymorphic overload of this same function name with updated signature
     * Reason: Legacy signature lacks support for dynamic quantization modes.
     * Removal target: v0.2.0 (Q2 2026)
     */
    InferEngine(
        const InfinilmModel::Config &config,
        const distributed::DistConfig &distributed_config = distributed::DistConfig(),
        infinicore::Device::Type device_type = infinicore::context::getDevice().getType(),
        const cache::CacheConfig *cache_config = nullptr,
        bool enable_graph_compiling = false);

    InferEngine(
        const std::string &model_path = "",
        const distributed::DistConfig &distributed_config = distributed::DistConfig(),
        infinicore::Device::Type device_type = infinicore::context::getDevice().getType(),
        const cache::CacheConfig *cache_config = nullptr,
        bool enable_graph_compiling = false);

    // Load a parameter to all workers (each can extract its shard inside RankWorker)
    void load_param(const std::string &name, const infinicore::Tensor &param);

    // return the parameters (i.e. weights and biases).
    std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> state_dict();

    // Run a single forward pass on all workers and return the outputs from all ranks
    Output forward(const Input &input);

    void compile();

    void reset_cache(const cache::CacheConfig *new_config);

    ~InferEngine();

    const distributed::DistConfig &get_dist_config() const;

    // Get current KV configuration
    const cache::CacheConfig *get_cache_config() const { return cache_config_.get(); }

protected:
    std::vector<std::unique_ptr<RankWorker>> workers_;
    std::unique_ptr<RankBarrier> barrier_;
    distributed::CommunicationGroup communication_group_;
    std::unique_ptr<cache::CacheConfig> cache_config_;
    const InfinilmModel::Config &legacy_model_config_ = InfinilmModel::Config();
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

} // namespace infinilm::engine
