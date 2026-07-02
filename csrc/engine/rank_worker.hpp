#pragma once

#include "../backends/attention_backends.hpp"
#include "../cache/cache.hpp"
#include "../config/model_config.hpp"
#include "../global_state/global_state.hpp"
#include "../models/model_factory.hpp"
#include "compiler/general_compiler.hpp"
#include "distributed/distributed.hpp"
#include "rank_barrier.hpp"

#include <any>
#include <condition_variable>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace infinilm::engine {

using ForwardContext = infinilm::global_state::ForwardContext;

class RankWorker {
    enum class Command {
        INIT,
        LOAD,
        LOAD_BATCH,
        PREPROCESS,
        RUN,
        RESET_CACHE,
        COMPILE,
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
        /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
        std::optional<infinicore::Tensor> input_offsets;
        /// Cumulative total sequence lengths for each request, of shape `[num_requests + 1]`.
        std::optional<infinicore::Tensor> cu_seqlens;
        /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
        std::optional<infinicore::Tensor> block_tables;
        /// Slot ids for each token `[seq]`. Used for paged cache.
        std::optional<infinicore::Tensor> slot_mapping;
        /// Image pixel values for multi-modal models.
        std::optional<std::vector<infinicore::Tensor>> pixel_values;
        /// Image placeholder bounds for MiniCPM-V style replacement.
        std::optional<std::vector<infinicore::Tensor>> image_bound;
        /// Target patch sizes for each image (MiniCPM-V).
        std::optional<std::vector<infinicore::Tensor>> tgt_sizes;
        /// req_id for each pixel_values among a batch
        std::optional<std::vector<size_t>> image_req_ids;
        /// Flattened [start, end) visual token ranges in the packed language sequence.
        std::optional<std::vector<size_t>> visual_token_ranges;

        float temperature{1};

        int top_k{50};

        float top_p{1};

        infinilm::InfinilmModel::Input to_model_input(infinicore::Device device) const;
    };

    struct Output {
        infinicore::Tensor output_ids;
    };

    RankWorker(std::shared_ptr<infinilm::global_state::InfinilmConfig> infinilm_config,
               const distributed::RankInfo &rank_info,
               const cache::CacheConfig *cache_config,
               RankBarrier *barrier,
               bool enable_graph_compiling,
               backends::AttentionBackend attention_backend);

    // Submit a parameter load job and wait until the load completes on the worker thread.
    void load_param(const std::string &name,
                    const infinicore::Tensor &param);

    void load_params(const std::unordered_map<std::string, infinicore::Tensor> &params, bool strict = true);

    void process_weights_after_loading();

    // return the parameters (i.e. weights and biases).
    std::unordered_map<std::string, infinicore::nn::Parameter> state_dict();

    std::vector<std::string> state_dict_keys();

    // Submit a run (forward) job.
    void run(const Input &args);

    // Reset the internal cache with a new configuration
    void reset_cache(const cache::CacheConfig *new_config);

    std::vector<infinicore::Tensor> get_kv_cache();

    // Compile the model graph if enabled.
    void compile();

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
    std::shared_ptr<infinilm::global_state::InfinilmConfig> infinilm_config_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    engine::distributed::RankInfo rank_info_;
    ForwardContext forward_context_;
    std::shared_ptr<InfinilmModel> model_;
    std::shared_ptr<cache::Cache> cache_;

    // Backends
    backends::AttentionBackend attention_backend_;

    // Graph Compiling
    bool enable_graph_compiling_;
    std::unique_ptr<GraphCompiler> compiler_;

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
    std::unordered_map<std::string, infinicore::Tensor> pending_params_;
    bool pending_params_strict_ = true;
    Input pending_args_;
    std::unique_ptr<cache::CacheConfig> pending_cache_config_;

    // Output (protected by mutex)
    Output output_;

    // Thread sync
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;

    // Random
    std::mt19937 rng_;

    RankBarrier *barrier_;
};

} // namespace infinilm::engine
