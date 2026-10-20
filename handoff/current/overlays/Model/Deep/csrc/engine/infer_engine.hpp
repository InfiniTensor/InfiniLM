#pragma once
#include "../weights/weights.hpp"

#include "../config/model_config.hpp"
#include "../global_state/global_state.hpp"
#include "../models/infinilm_model.hpp"
#include "distributed/distributed.hpp"
#include "rank_barrier.hpp"
#include "rank_worker.hpp"
#include "runtime/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infinilm::engine {

class InferEngine {
public:
    using Input = RankWorker::Input;

    using Output = RankWorker::Output;

    // Updated constructor: accept CacheConfig instead of CacheType
    InferEngine(
        const std::string &config_str,
        const distributed::DistConfig &distributed_config,
        infinilm::runtime::Device::Type device_type,
        const cache::CacheConfig *cache_config = nullptr,
        bool enable_graph_compiling = false,
        backends::AttentionBackend attention_backend = backends::AttentionBackend::Default,
        std::optional<infinilm::runtime::DataType> kv_cache_dtype = std::nullopt,
        bool use_mla = false,
        const std::string &weight_load_mode = "async",
        bool pre_transpose = false);

    // Load a parameter to all workers (each can extract its shard inside RankWorker)
    void load_param(const std::string &name, const infinilm::runtime::Tensor &param);

    // Load a batch of parameters to all workers, syncing each worker once after the batch.
    void load_params(const std::unordered_map<std::string, infinilm::runtime::Tensor> &params, bool strict = true);

    // process the weights after loading on all workers (e.g., for quantization)
    void process_weights_after_loading();

    // return the parameters (i.e. weights and biases).
    std::vector<std::unordered_map<std::string, infinilm::weights::Weight>> state_dict();

    std::vector<std::string> state_dict_keys();

    // Run a single forward pass on all workers and return sampled token IDs.
    Output forward(const Input &input);

    void compile();

    void reset_cache(const cache::CacheConfig *new_config);

    std::vector<std::vector<infinilm::runtime::Tensor>> get_kv_cache();
    // Readonly real VideoNSA raw-pool lifetime; never a tensor pointer.
    std::uint64_t get_nsa_pool_epoch();

    ~InferEngine();

    const distributed::DistConfig &get_dist_config() const;

    // Get current KV configuration
    std::unique_ptr<cache::CacheConfig> get_cache_config() const;

protected:
    enum class WeightState { Loading,
                             Finalizing,
                             Ready,
                             Failed };
    void compile_workers();

    std::vector<std::unique_ptr<RankWorker>> workers_;
    std::unique_ptr<RankBarrier> barrier_;
    distributed::CommunicationGroup communication_group_;
    std::unique_ptr<cache::CacheConfig> cache_config_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    backends::AttentionBackend attention_backend_ = backends::AttentionBackend::Default;
    std::string weight_load_mode_ = "async";
    WeightState weight_state_ = WeightState::Loading;
    // An untouched engine is still used by the explicit skip_load test mode.
    // Once loading starts, a partial model may not execute.
    bool weight_load_started_ = false;
    std::unordered_set<std::string> loaded_weight_keys_;
    // A model instance submits one load, compile, cache reset or forward at a
    // time. RankWorker queues are not reentrant for concurrent callers.
    mutable std::mutex submission_mutex_;
    bool use_mla_{false};
    std::uint64_t nsa_pool_epoch_{0};
    std::vector<size_t> native_sparse_lengths_;
};

} // namespace infinilm::engine
