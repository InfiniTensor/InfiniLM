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
        STOP
    };

public:
    struct Input {
        /// Token IDs tensor of shape `[batch, seq_len]`.
        std::optional<infinicore::Tensor> input_ids;
        /// Position IDs tensor of shape `[batch, seq_len]` or `[seq_len]`.
        std::optional<infinicore::Tensor> position_ids;
        /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> past_sequence_lengths;
        /// ToTal Lengths for each request sequence, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> total_sequence_lengths;
        /// Offsets of each request in a continous-batched sequence, of shape `[num_requests]`.
        std::optional<infinicore::Tensor> input_offsets;
        /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
        std::optional<infinicore::Tensor> block_tables;
        /// Slot ids for each token `[seq]`. Used for paged cache.
        std::optional<infinicore::Tensor> slot_mapping;

        float temperature{1};

        int top_k{50};

        float top_p{1};

        float random_val{0.1};

        infinilm::InfinilmModel::Input to_model_input(infinicore::Device device) const;
    };

    struct Output {
        infinicore::Tensor output_ids;
    };

    RankWorker(const InfinilmModel::Config &model_config,
               const distributed::RankInfo &rank_info,
               const cache::CacheConfig *cache_config);

    // Submit a parameter load job and wait until the load completes on the worker thread.
    void load_param(const std::string &name,
                    const infinicore::Tensor &param);

    // return the parameters (i.e. weights and biases).
    std::unordered_map<std::string, infinicore::nn::Parameter> state_dict();

    // Submit a run (forward) job.
    void run(const Input &args);

    // Reset the internal cache with a new configuration
    void reset_cache(const cache::CacheConfig *new_config);

    // Wait until run job completes. The result can be retrieved with get_output().
    void wait();

    // Request worker shutdown and join the thread.
    void close();

    // Thread-safe accessor for last output produced by RUN.
    Output get_output();

    std::string info() const;

private:
    void thread_loop();

private:
    // Worker properties
    const InfinilmModel::Config &model_config_;
    distributed::RankInfo rank_info_;
    std::shared_ptr<InfinilmModel> model_;
    std::shared_ptr<cache::Cache> cache_;

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
    Input pending_args_;
    std::unique_ptr<cache::CacheConfig> pending_cache_config_;

    // Output (protected by mutex)
    Output output_;

    // Thread sync
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

} // namespace infinilm::engine
