#pragma once

#include "../cache/cache.hpp"
#include "../models/model_factory.hpp"
#include "distributed/distributed.hpp"

#include <any>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace infinilm::engine {

class RankWorker {
    enum class Command {
        INIT,
        LOAD,
        RUN,
        RESET_CACHE,
        RESET_CACHE_WITH_CONFIG,
        STOP
    };

public:
    RankWorker(const std::any &model_config,
               const distributed::RankInfo &rank_info,
               const cache::CacheConfig &cache_config);

    // Submit a parameter load job and wait until the load completes on the worker thread.
    void load_param(const std::string &name,
                    const infinicore::Tensor &param);

    // return the parameters (i.e. weights and biases).
    std::unordered_map<std::string, infinicore::nn::Parameter> state_dict();

    // Submit a run (forward) job.
    void run(const std::vector<std::any> &args);

    // Reset the internal cache in the model (clears state between generations)
    void reset_cache(size_t pos = 0);

    // Reset the internal cache with a new configuration
    void reset_cache(const cache::CacheConfig &new_config, size_t pos = 0);

    // Wait until run job completes. The result can be retrieved with get_output().
    void wait();

    // Request worker shutdown and join the thread.
    void close();

    // Thread-safe accessor for last output produced by RUN.
    infinicore::Tensor get_output();

    std::string info() const;

private:
    void thread_loop();

private:
    // Worker properties
    std::any model_config_;
    distributed::RankInfo rank_info_;
    std::shared_ptr<InfinilmModel> model_;
    std::shared_ptr<cache::CacheInterface> cache_ptr_;

    // Command for the pending job (protected by mutex_)
    Command job_cmd_;

    // State flags (protected by mutex_)
    bool has_job_ = false;     // a job is pending
    bool job_done_ = false;    // last job completed
    bool should_exit_ = false; // request to stop
    bool init_done_ = false;   // initialization finished

    // Task payloads (protected by mutex)
    std::string pending_param_name_;
    infinicore::Tensor pending_param_;
    std::vector<std::any> pending_args_;
    size_t pending_reset_pos_ = 0;
    cache::CacheConfig pending_cache_config_;

    // Output (protected by mutex)
    infinicore::Tensor output_;

    // Thread sync
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

} // namespace infinilm::engine
