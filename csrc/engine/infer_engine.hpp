#pragma once

#include "../config/model_config.hpp"
#include "../global_state/global_state.hpp"
#include "../models/infinilm_model.hpp"
#include "distributed/distributed.hpp"
#include "infinicore/tensor.hpp"
#include "rank_barrier.hpp"
#include "rank_worker.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace infinilm::engine {

class InferEngine {
public:
    using Input = RankWorker::Input;

    using Output = RankWorker::Output;

    // Updated constructor: accept CacheConfig instead of CacheType
    InferEngine(
        const std::string &config_str,
        const distributed::DistConfig &distributed_config = distributed::DistConfig(),
        infinicore::Device::Type device_type = infinicore::context::getDevice().getType(),
        const cache::CacheConfig *cache_config = nullptr,
        bool enable_graph_compiling = false,
        backends::AttentionBackend attention_backend = backends::AttentionBackend::Default,
        std::optional<infinicore::DataType> kv_cache_dtype = std::nullopt,
        bool use_mla = false,
        const std::string &weight_load_mode = "async");

    // Load a parameter to all workers (each can extract its shard inside RankWorker)
    void load_param(const std::string &name, const infinicore::Tensor &param);

    // Load a batch of parameters to all workers, syncing each worker once after the batch.
    void load_params(const std::unordered_map<std::string, infinicore::Tensor> &params, bool strict = true);

    // process the weights after loading on all workers (e.g., for quantization)
    void process_weights_after_loading();

    // return the parameters (i.e. weights and biases).
    std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> state_dict();

    std::vector<std::string> state_dict_keys();

    // Run a single forward pass on all workers and return the outputs from all ranks
    Output forward(const Input &input);

    void compile();

    void reset_cache(const cache::CacheConfig *new_config);

    std::vector<std::vector<infinicore::Tensor>> get_kv_cache();

    ~InferEngine();

    const distributed::DistConfig &get_dist_config() const;

    // Get current KV configuration
    const cache::CacheConfig *get_cache_config() const { return cache_config_.get(); }

protected:
    std::vector<std::unique_ptr<RankWorker>> workers_;
    std::unique_ptr<RankBarrier> barrier_;
    distributed::CommunicationGroup communication_group_;
    std::unique_ptr<cache::CacheConfig> cache_config_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    backends::AttentionBackend attention_backend_ = backends::AttentionBackend::Default;
    std::string weight_load_mode_ = "async";
    bool weights_finalized_ = false;
    bool use_mla_{false};
};

} // namespace infinilm::engine
