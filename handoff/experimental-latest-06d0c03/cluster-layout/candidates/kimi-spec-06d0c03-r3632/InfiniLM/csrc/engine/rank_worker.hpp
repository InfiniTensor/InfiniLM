#pragma once
#include "../weights/weights.hpp"

#include "../backends/attention_backends.hpp"
#include "../cache/cache.hpp"
#include "../config/model_config.hpp"
#include "../global_state/global_state.hpp"
#include "../models/model_factory.hpp"
#include "compiler/graph_compiler.hpp"
#include "compiler/paged_decode_bucket.hpp"
#include "distributed/distributed.hpp"
#include "nsa_owner_input.hpp"
#include "rank_barrier.hpp"
#include "static_decode_metadata.hpp"

#include <any>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
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
        VALIDATE_W4,
        PREPARE_W4,
        COMMIT_W4,
        RUN,
        RESET_CACHE,
        COMPILE,
        STOP
    };

public:
    struct Input {
        /// Token IDs tensor of shape `[batch, seq_len]`.
        std::optional<infinilm::runtime::Tensor> input_ids;
        /// Position IDs tensor of shape `[batch, seq_len]` or `[seq_len]`.
        std::optional<infinilm::runtime::Tensor> position_ids;
        /// Past Lengths of cached sequence for each request, of shape `[num_requests]`.
        std::optional<infinilm::runtime::Tensor> past_sequence_lengths;
        /// ToTal Lengths for each request sequence, of shape `[num_requests]`.
        std::optional<infinilm::runtime::Tensor> total_sequence_lengths;
        /// Offsets of each request in a continous-batched sequence, of shape `[num_requests + 1]`.
        std::optional<infinilm::runtime::Tensor> input_offsets;
        /// Cumulative total sequence lengths for each request, of shape `[num_requests + 1]`.
        std::optional<infinilm::runtime::Tensor> cu_seqlens;
        /// Block ids for each request `[batch, max_block_table_length]`. Used for paged cache.
        std::optional<infinilm::runtime::Tensor> block_tables;
        /// Slot ids for each token `[seq]`. Used for paged cache.
        std::optional<infinilm::runtime::Tensor> slot_mapping;
        /// Mamba state cache indices read at the start of each request forward.
        std::optional<infinilm::runtime::Tensor> mamba_init_state_indices;
        /// Mamba state cache indices written with the final state of each request forward.
        std::optional<infinilm::runtime::Tensor> mamba_final_state_indices;
        /// Image pixel values for multi-modal models.
        std::optional<std::vector<infinilm::runtime::Tensor>> pixel_values;
        /// Image placeholder bounds for MiniCPM-V style replacement.
        std::optional<std::vector<infinilm::runtime::Tensor>> image_bound;
        /// Target patch sizes for each image (MiniCPM-V).
        std::optional<std::vector<infinilm::runtime::Tensor>> tgt_sizes;
        /// Qwen-style image grids. Vector of tensors shape: [3] with temporal, height, width.
        std::optional<std::vector<infinilm::runtime::Tensor>> image_grid_thw;
        /// req_id for each pixel_values among a batch
        std::optional<std::vector<size_t>> image_req_ids;
        /// Flattened [start, end) visual token ranges in the packed language sequence.
        std::optional<std::vector<size_t>> visual_token_ranges;
        /// Target model hidden states for draft/MTP models.
        std::optional<infinilm::runtime::Tensor> target_hidden_states;
        /// Sample logits at every packed input position instead of one token per request.
        bool sample_all_positions{false};

        float temperature{1};

        int top_k{50};

        float top_p{1};

        // Optional request-owned draws, one per sampled request. Direct engine
        // callers may omit this while the legacy API is being retired.
        std::optional<std::vector<float>> random_values;

        // Sampling parameters in request order. The scalar fields above remain
        // the direct-engine default when this optional batch input is absent.
        std::optional<std::vector<std::tuple<float, float, int>>> sampling_rows;

        std::optional<nsa::Snapshot> nsa_owner_snapshot;
        bool kimi_grouped_prefill{false};
        bool kimi_tiled_w13_r1861{false};
        bool kimi_owned_indices_r1213{false};
        bool kimi_identity_indices_r1213{false};
        bool defer_pipeline_feedback{false};

        infinilm::InfinilmModel::Input to_model_input(
            infinilm::runtime::Device device,
            ForwardContext &forward_context,
            bool snapshot_static_sequence_lengths = false,
            bool preserve_target_hidden_device = false,
            size_t logical_kv_upper_bound = 0,
            std::shared_ptr<const nsa::Proof> nsa_owner = {}) const;
    };

    struct Output {
        infinilm::runtime::Tensor output_ids;
        infinilm::runtime::Tensor logits;
        infinilm::runtime::Tensor hidden_states;
        // Present only after this forward actually submitted the selected plan.
        std::optional<infinilm::execution::graph::Graph::ExecutionReport> graph_execution;
        const char *eager_reason = "graphs_disabled";
    };

    RankWorker(std::shared_ptr<infinilm::global_state::InfinilmConfig> infinilm_config,
               const distributed::RankInfo &rank_info,
               const cache::CacheConfig *cache_config,
               RankBarrier *barrier,
               bool enable_graph_compiling,
               backends::AttentionBackend attention_backend,
               std::uint64_t nsa_pool_epoch = 0);

    void wait_for_init();

    ~RankWorker();

    // Submit a parameter load job and wait until the load completes on the worker thread.
    void load_param(const std::string &name,
                    const infinilm::runtime::Tensor &param);

    void load_params(const std::unordered_map<std::string, infinilm::runtime::Tensor> &params, bool strict = true);

    void process_weights_after_loading();
    void validate_w4_weights();
    void prepare_w4_weights();
    void commit_w4_weights();

    // return the parameters (i.e. weights and biases).
    std::unordered_map<std::string, infinilm::weights::Weight> state_dict();

    std::vector<std::string> state_dict_keys();
    std::vector<std::string> state_required_keys();

    // Submit a run (forward + sampling) job.
    void run(const Input &args);

    // Reset the internal cache with a new configuration
    void reset_cache(const cache::CacheConfig *new_config, std::uint64_t nsa_pool_epoch = 0);
    std::uint64_t get_nsa_pool_epoch();

    std::vector<infinilm::runtime::Tensor> get_kv_cache();

    // Compile the model graph if enabled.
    void compile();

    // Wait until run job completes. The result can be retrieved with get_output().
    void wait();

    // Request worker shutdown and join the thread.
    void close();
    void request_stop() noexcept;
    void join() noexcept;

    // Thread-safe accessor for last output produced by RUN.
    Output get_output();

    std::string info() const;

private:
    void thread_loop();
    void submit_w4_phase(Command command);
    void fail(const char *operation, std::exception_ptr error) noexcept;
    void refresh_finite_paged_decode_profile();
    bool drain_nsa_owner_noexcept() noexcept;

private:
    // Worker properties
    std::shared_ptr<infinilm::global_state::InfinilmConfig> infinilm_config_;
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
    engine::distributed::RankInfo rank_info_;
    // Created on the worker thread and retained until its model and plans die.
    std::shared_ptr<infinilm::runtime::Runtime> runtime_;
    ForwardContext forward_context_;
    std::uint64_t nsa_run_serial_{0};
    std::uint64_t nsa_pool_epoch_{0}, pending_nsa_pool_epoch_{0};
    // CPU validity revocation never clears this backing retirement obligation.
    bool nsa_drain_required_{false}, nsa_terminal_undrained_{false};
    std::shared_ptr<InfinilmModel> model_;
    // Declared after model/runtime; retained owners retire before those members.
    std::unique_ptr<quantization::W4ModelPlan> w4_plan_;
    std::shared_ptr<cache::Cache> cache_;

    // Backends
    backends::AttentionBackend attention_backend_;

    // Graph Compiling
    bool enable_graph_compiling_;
    std::unique_ptr<GraphCompiler> compiler_;
    StaticDecodeMetadata static_decode_metadata_;
    // Worker-owned cache configuration; refreshed only after model/cache creation.
    std::optional<FinitePagedDecodeProfile> finite_paged_decode_profile_;

    // Command for the pending job (protected by mutex_)
    Command job_cmd_;

    // State flags (protected by mutex_)
    bool has_job_ = false;       // a job is pending
    bool job_done_ = false;      // last job completed
    bool should_exit_ = false;   // request to stop
    bool init_done_ = false;     // initialization finished
    std::exception_ptr failure_; // first worker failure, protected by mutex_

    // Task payloads (protected by mutex)
    std::string pending_param_name_;
    infinilm::runtime::Tensor pending_param_;
    std::unordered_map<std::string, infinilm::runtime::Tensor> pending_params_;
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
