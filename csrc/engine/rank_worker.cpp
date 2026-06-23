#include "rank_worker.hpp"

#include "../global_state/ar_profile.hpp"
#include "../global_state/hang_trace.hpp"
#include "../global_state/global_state.hpp"
#include "../utils/agent_debug.hpp"
#include "../models/model_factory.hpp"
#include "../models/models_registry.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::engine {

namespace {

bool rank_worker_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_RANK_WORKER_PROFILE");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

/// Keep pre-graph RankBarrier on decode replay (default on for TP rank sync).
/// Opt out for perf bisect: INFINI_DECODE_SKIP_PRE_BARRIER=1
bool decode_skip_pre_barrier() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_DECODE_SKIP_PRE_BARRIER");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

/// Event-based post-AR wait instead of full stream sync (default off).
/// Bisect: INFINI_DECODE_LIGHT_SYNC=1
bool decode_light_sync_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_DECODE_LIGHT_SYNC");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

bool row_is_final_prefill_chunk(const std::vector<bool> &flags, size_t row) {
    return flags.empty() || row >= flags.size() || flags[row];
}

bool any_final_prefill_chunk(const std::vector<bool> &flags) {
    return infinilm::InfinilmModel::any_final_prefill_chunk(flags);
}

} // namespace

RankWorker::RankWorker(
    std::shared_ptr<infinilm::global_state::InfinilmConfig> infinilm_config,
    const distributed::RankInfo &rank_info,
    const cache::CacheConfig *cache_config,
    RankBarrier *barrier,
    bool enable_graph_compiling,
    backends::AttentionBackend attention_backend)
    : infinilm_config_(infinilm_config),
      model_config_(infinilm_config->model_config),
      rank_info_(rank_info),
      attention_backend_(attention_backend),
      enable_graph_compiling_(enable_graph_compiling),
      job_cmd_(Command::INIT),
      has_job_(false),
      job_done_(false),
      should_exit_(false),
      init_done_(false),
      rng_(std::random_device{}()),
      barrier_(barrier) {
    if (cache_config != nullptr) {
        pending_cache_config_ = cache_config->unique_copy();
    }
    // start the thread
    thread_ = std::thread(&RankWorker::thread_loop, this);
    // Wait until the worker thread finishes initialization (model created)
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_; });
}

std::string RankWorker::info() const {
    std::stringstream ss;

    ss << "RankWorker{";

    // Rank related
    ss << rank_info_.to_string() << " ";

    // Flags
    ss << "| init_done: " << (init_done_ ? "true" : "false") << " ";
    ss << "| should_exit: " << (should_exit_ ? "true" : "false") << " ";
    ss << "| has_job: " << (has_job_ ? "true" : "false") << " ";
    ss << "| job_done: " << (job_done_ ? "true" : "false") << " ";

    ss << "}";

    return ss.str();
}

bool RankWorker::has_failed() {
    std::lock_guard<std::mutex> lock(mutex_);
    return should_exit_;
}

void RankWorker::signal_should_exit() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        should_exit_ = true;
        job_done_ = true;
    }
    cv_.notify_all();
}

//------------------------------------------------------
// load_param -- synchronous (blocks until worker finishes loading)
//------------------------------------------------------
void RankWorker::load_param(const std::string &name,
                            const infinicore::Tensor &param) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // If the worker is stopping, don't submit new jobs.
        if (should_exit_) {
            throw std::runtime_error("RankWorker is closing; cannot load_param");
        }

        pending_param_name_ = name;
        pending_param_ = param;

        job_cmd_ = Command::LOAD;
        has_job_ = true;
        job_done_ = false;
    }
    cv_.notify_all();

    // Wait for job completion
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return job_done_ || should_exit_; });

    if (should_exit_) {
        throw std::runtime_error("RankWorker stopped while loading parameter");
    }
}

//------------------------------------------------------
// process_weights_after_loading -- asynchronous
//------------------------------------------------------
void RankWorker::process_weights_after_loading() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // If the worker is stopping, don't submit new jobs.
        if (should_exit_) {
            throw std::runtime_error("RankWorker is closing; cannot process_weights_after_loading");
        }

        job_cmd_ = Command::PREPROCESS;
        has_job_ = true;
        job_done_ = false;
    }
    cv_.notify_all();

    // Wait for job completion
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return job_done_ || should_exit_; });

    if (should_exit_) {
        throw std::runtime_error("RankWorker stopped while processing weights");
    }
}

//------------------------------------------------------
// state_dict --
//------------------------------------------------------
std::unordered_map<std::string, infinicore::nn::Parameter> RankWorker::state_dict() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });

    if (!model_) {
        throw std::runtime_error("state_dict called before model initialization");
    }

    return model_->state_dict();
}

//------------------------------------------------------
// run -- asynchronous
//------------------------------------------------------
void RankWorker::run(const Input &args) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (should_exit_) {
        throw std::runtime_error("RankWorker is closing; cannot run");
    }

    pending_args_ = args;
    job_cmd_ = Command::RUN;
    has_job_ = true;
    job_done_ = false;

    cv_.notify_all();
}

//------------------------------------------------------
// compile -- asynchronous
//------------------------------------------------------
void RankWorker::compile() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (should_exit_) {
        throw std::runtime_error("RankWorker is closing; cannot run");
    }

    job_cmd_ = Command::COMPILE;
    has_job_ = true;
    job_done_ = false;
    cv_.notify_all();
}

//------------------------------------------------------
// wait -- asynchronous
//------------------------------------------------------
void RankWorker::wait() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return job_done_ || should_exit_; });

    if (should_exit_) {
        throw std::runtime_error("RankWorker stopped during run");
    }
}

void RankWorker::reset_cache(const cache::CacheConfig *new_config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (should_exit_) {
        throw std::runtime_error("RankWorker is closing; cannot reset_cache");
    }

    // Store both the position and the new config
    pending_cache_config_ = new_config->unique_copy();
    job_cmd_ = Command::RESET_CACHE;
    has_job_ = true;
    job_done_ = false;
    cv_.notify_all();
}

//------------------------------------------------------
// close -- request shutdown and join thread
//------------------------------------------------------
void RankWorker::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        should_exit_ = true;
        has_job_ = false; // don't keep old jobs pending
        job_cmd_ = Command::STOP;
    }
    cv_.notify_all();

    if (thread_.joinable()) {
        thread_.join();
    }
}

//------------------------------------------------------
// get_output (thread safe)
//------------------------------------------------------
RankWorker::Output RankWorker::get_output() {
    std::lock_guard<std::mutex> lock(mutex_);
    return output_;
}

std::vector<infinicore::Tensor> RankWorker::get_paged_kv_cache_tensors() {
    std::lock_guard<std::mutex> lock(mutex_);
    return kv_cache_snapshot_;
}

PagedCompiler::GraphStats RankWorker::prefill_graph_stats() const {
    if (compiler_ == nullptr) {
        return {};
    }
    if (auto *general = dynamic_cast<GeneralCompiler *>(compiler_.get())) {
        return general->graph_stats();
    }
    return {};
}

std::vector<size_t> RankWorker::native_capture_buckets() const {
    if (compiler_ == nullptr) {
        return {};
    }
    if (auto *general = dynamic_cast<GeneralCompiler *>(compiler_.get())) {
        return general->native_capture_buckets();
    }
    return {};
}

void RankWorker::decode_post_ar_sync() {
    if (decode_light_sync_enabled()) {
        infinicore::context::recordEvent(decode_ar_event_);
        infinicore::context::streamWaitEvent(infinicore::context::getStream(), decode_ar_event_);
        if (rank_info_.tp_rank == 0) {
            infinicore::context::synchronizeEvent(decode_ar_event_);
        }
    } else {
        infinicore::context::syncStream();
    }
}

//------------------------------------------------------
// thread_loop
//------------------------------------------------------
void RankWorker::thread_loop() {
    try {
        {
            std::lock_guard<std::mutex> lk(mutex_);

            // Initialize device & model outside of holding the main mutex to avoid blocking callers.
            infinicore::context::setDevice(rank_info_.device);
            decode_ar_event_ = infinicore::context::createEvent();

            // Initialize global enviromnet.
            infinilm::global_state::initialize_model_parallel(rank_info_);
            infinilm::global_state::initialize_forward_context(forward_context_);
            infinilm::global_state::initialize_infinilm_config(infinilm_config_);

            // Create model using factory (may be expensive)
            const std::string &model_type = model_config_->get<std::string>("model_type");
            const auto &model_map = models::get_causal_lm_model_map();
            auto it = model_map.find(model_type);
            if (it != model_map.end()) {
                model_ = InfinilmModelFactory::createModel(
                    model_config_,
                    rank_info_.device,
                    pending_cache_config_ != nullptr ? pending_cache_config_.get() : nullptr);
            } else {
                std::vector<std::string> classic_models = {"llama", "minicpm", "fm9g", "fm9g7b"};
                if ((std::find(classic_models.begin(), classic_models.end(), model_type) != classic_models.end())) {
                    model_ = InfinilmModelFactory::createModel(
                        model_config_,
                        rank_info_,
                        pending_cache_config_ != nullptr ? pending_cache_config_.get() : nullptr,
                        attention_backend_);
                } else {
                    throw std::runtime_error("RankWorker::thread_loop(): Unsupported model config type: " + model_type);
                }
            }

            if (!model_) {
                throw std::runtime_error("Failed to create model");
            }
            kv_cache_snapshot_ = global_state::get_forward_context().kv_cache_vec;
            if (enable_graph_compiling_) {
                compiler_ = std::make_unique<GeneralCompiler>(model_, barrier_);
            }

            init_done_ = true;
        }
        cv_.notify_all();

        // Main loop: wait for jobs or exit
        while (true) {
            Command local_cmd = Command::INIT;
            std::string local_param_name;
            infinicore::Tensor local_param;
            Input local_args;
            std::unique_ptr<cache::CacheConfig> local_cache_config;

            // Wait for a job or exit
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_.wait(lk, [&] { return has_job_ || should_exit_; });

                if (should_exit_) {
                    break;
                }

                // capture job data and clear has_job_
                local_cmd = job_cmd_;
                if (local_cmd == Command::LOAD) {
                    local_param_name = pending_param_name_;
                    local_param = pending_param_;
                } else if (local_cmd == Command::PREPROCESS) {

                } else if (local_cmd == Command::RUN) {
                    local_args = pending_args_;
                } else if (local_cmd == Command::RESET_CACHE) {
                    if (pending_cache_config_ != nullptr) {
                        local_cache_config = pending_cache_config_->unique_copy();
                    }
                }
                // mark job as being processed
                has_job_ = false;
                job_done_ = false;
            } // unlock mutex while executing the job

            // Execute job outside the lock
            if (local_cmd == Command::LOAD) {
                try {
                    model_->load_parameter(local_param_name, local_param);
                    // #region agent log
                    if (local_param_name.find("self_attn.q_proj.weight") != std::string::npos
                        && local_param_name.find("layers.0.") != std::string::npos) {
                        const auto &sd = model_->state_dict();
                        auto it = sd.find(local_param_name);
                        if (it != sd.end()) {
                            const auto &stored = it->second;
                            const auto shape = stored->shape();
                            auto cpu = stored->to(infinicore::Device::cpu());
                            const auto *data = reinterpret_cast<const uint16_t *>(cpu->data());
                            std::string shape_str = "[";
                            for (size_t i = 0; i < shape.size(); ++i) {
                                if (i > 0) {
                                    shape_str += ",";
                                }
                                shape_str += std::to_string(shape[i]);
                            }
                            shape_str += "]";
                            infinilm::agent_debug::log(
                                "rank_worker.cpp:LOAD",
                                "q_proj_layer0_stored_param",
                                "M",
                                std::string("{\"tp_rank\":") + std::to_string(rank_info_.tp_rank) +
                                    ",\"tp_size\":" + std::to_string(rank_info_.tp_size) +
                                    ",\"input_numel\":" + std::to_string(local_param->numel()) +
                                    ",\"stored_numel\":" + std::to_string(stored->numel()) +
                                    ",\"stored_shape\":\"" + shape_str + "\"" +
                                    ",\"first_fp16_bits\":" + std::to_string(data[0]) + "}",
                                "post-fix");
                        }
                    }
                    // #endregion
                } catch (const std::exception &e) {
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        should_exit_ = true;
                        job_done_ = true;
                    }
                    cv_.notify_all();
                    spdlog::error("[{}] exception during load_parameter_: {}\n", info(), e.what());
                    break;
                }

                // signal completion
                {
                    std::lock_guard<std::mutex> lk(mutex_);
                    job_done_ = true;
                }
                cv_.notify_all();

            } else if (local_cmd == Command::PREPROCESS) {
                // Handle preprocess command
                try {
                    model_->process_weights_after_loading();
                } catch (const std::exception &e) {
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        should_exit_ = true;
                        job_done_ = true;
                    }
                    cv_.notify_all();
                    spdlog::error("[{}] exception during process_weights_after_loading_: {}\n", info(), e.what());
                    break;
                }

                // signal completion
                {
                    std::lock_guard<std::mutex> lk(mutex_);
                    job_done_ = true;
                }
                cv_.notify_all();
            } else if (local_cmd == Command::RUN) {
                try {
                    // TP ranks can begin processing RUN ms apart (thread scheduling);
                    // sync before any compiler/graph work so collectives stay aligned.
                    if (rank_info_.tp_size > 1) {
                        barrier_->wait("run_job_tp_sync", rank_info_.tp_rank);
                    }
                    global_state::hang_trace::ScopedBracket run_job_bracket(
                        "run_job", rank_info_.tp_rank);
                    {
                        std::lock_guard<std::mutex> lk(mutex_);

                        infinicore::Tensor logits;
                        bool is_prefill = false;
                        bool is_mixed = false;
                        bool skip_sampling = false;
                        bool piecewise_ran = false;
                        size_t n_req = 1;
                        size_t batch_size = 0;
                        const char *mode_str = "unknown";
                        if (local_args.scheduling_mode.has_value()) {
                            switch (*local_args.scheduling_mode) {
                            case SchedulingMode::PREFILL:
                                is_prefill = true;
                                mode_str = "PREFILL";
                                break;
                            case SchedulingMode::DECODE:
                                is_prefill = false;
                                mode_str = "DECODE";
                                break;
                            case SchedulingMode::MIXED:
                                is_mixed = true;
                                is_prefill = false;
                                mode_str = "MIXED";
                                break;
                            }
                        } else if (local_args.block_tables.has_value()
                                   && local_args.input_ids.has_value()) {
                            const size_t bs = local_args.block_tables.value()->size(0);
                            batch_size = bs;
                            const size_t input_width = local_args.input_ids.value()->size(1);
                            is_prefill = bs != input_width;
                            mode_str = is_prefill ? "PREFILL" : "DECODE";
                        }
                        if (local_args.block_tables.has_value()) {
                            batch_size = local_args.block_tables.value()->size(0);
                        }
                        if (local_args.input_offsets.has_value()) {
                            n_req = local_args.input_offsets.value()->size(0) - 1;
                        }
                        if (global_state::hang_trace::enabled() && rank_info_.tp_rank == 0) {
                            spdlog::info(
                                "hang_trace: run_job_begin mode={} n_req={} batch_size={} is_mixed={}",
                                mode_str,
                                n_req,
                                batch_size,
                                is_mixed);
                        }
                        auto model_input = local_args.to_model_input(infinicore::Device::cpu());
                        if (compiler_ != nullptr && !is_mixed) {
                            auto *general_compiler = dynamic_cast<GeneralCompiler *>(compiler_.get());
                            if (is_prefill && general_compiler != nullptr
                                && general_compiler->native_piecewise_enabled()) {
                                const bool profile = rank_worker_profile_enabled();
                                const double t_pw0 = profile ? monotonic_ms() : 0.0;
                                global_state::hang_trace::ScopedBracket pw_bracket(
                                    "native_piecewise", rank_info_.tp_rank);
                                if (profile) {
                                    size_t n_req = 1;
                                    size_t input_width = 0;
                                    if (local_args.block_tables.has_value()) {
                                        n_req = local_args.block_tables.value()->size(0);
                                    }
                                    if (local_args.input_ids.has_value()) {
                                        input_width = local_args.input_ids.value()->size(1);
                                    }
                                    const char *mode_str = "unknown";
                                    if (local_args.scheduling_mode.has_value()) {
                                        switch (*local_args.scheduling_mode) {
                                        case SchedulingMode::PREFILL:
                                            mode_str = "PREFILL";
                                            break;
                                        case SchedulingMode::DECODE:
                                            mode_str = "DECODE";
                                            break;
                                        case SchedulingMode::MIXED:
                                            mode_str = "MIXED";
                                            break;
                                        }
                                    }
                                    spdlog::info(
                                        "rank_worker_profile: RUN native_piecewise begin mode={} n_req={} "
                                        "input_width={} is_mixed={}",
                                        mode_str,
                                        n_req,
                                        input_width,
                                        is_mixed);
                                }
                                if (auto pw_logits = general_compiler->run_native_piecewise_prefill(model_input)) {
                                    logits = *pw_logits;
                                    piecewise_ran = true;
                                } else if (!any_final_prefill_chunk(model_input.is_final_prefill_chunk)) {
                                    piecewise_ran = true;
                                    skip_sampling = true;
                                }
                                if (profile) {
                                    spdlog::info(
                                        "rank_worker_profile: RUN native_piecewise end piecewise_ran={} "
                                        "skip_sampling={} wall_ms={:.3f}",
                                        piecewise_ran,
                                        skip_sampling,
                                        monotonic_ms() - t_pw0);
                                }
                            }
                            if (!logits && !piecewise_ran) {
                                auto [graph, output] = compiler_->get_compiled(model_input);
                                if (graph != nullptr && output != nullptr) {
                                    const bool ar_profile = global_state::ar_profile::enabled();
                                    const size_t graph_batch_size =
                                        model_input.block_tables.has_value()
                                            ? model_input.block_tables.value()->size(0)
                                            : 0;
                                    const size_t n_ar =
                                        global_state::get_forward_context().post_graph_allreduces.size();
                                    double pre_barrier_ms = 0.0;
                                    double graph_run_ms = 0.0;
                                    double post_sync_ms = 0.0;
                                    if (rank_info_.tp_size > 1 && !decode_skip_pre_barrier()) {
                                        const double t_barrier0 = ar_profile ? monotonic_ms() : 0.0;
                                        barrier_->wait("decode_pre_barrier", rank_info_.tp_rank);
                                        if (ar_profile) {
                                            pre_barrier_ms = monotonic_ms() - t_barrier0;
                                            global_state::ar_profile::counters().decode_pre_barrier_ms += pre_barrier_ms;
                                        }
                                    }
                                    {
                                        global_state::hang_trace::ScopedBracket graph_bracket(
                                            "decode_graph_run", rank_info_.tp_rank);
                                        const double t_graph0 = ar_profile ? monotonic_ms() : 0.0;
                                        graph->run();
                                        if (ar_profile) {
                                            graph_run_ms = monotonic_ms() - t_graph0;
                                            global_state::ar_profile::counters().decode_graph_run_ms += graph_run_ms;
                                        }
                                    }
                                    global_state::run_deferred_allreduces(
                                        global_state::get_forward_context().post_graph_allreduces);
                                    global_state::get_forward_context().post_graph_allreduces.clear();
                                    if (rank_info_.tp_size > 1) {
                                        global_state::hang_trace::ScopedBracket sync_bracket(
                                            "decode_post_sync", rank_info_.tp_rank);
                                        const double t_sync0 = ar_profile ? monotonic_ms() : 0.0;
                                        decode_post_ar_sync();
                                        if (ar_profile) {
                                            post_sync_ms = monotonic_ms() - t_sync0;
                                            global_state::ar_profile::counters().decode_post_sync_ms += post_sync_ms;
                                        }
                                    }
                                    if (ar_profile && rank_info_.tp_rank == 0) {
                                        global_state::ar_profile::counters().decode_graph_steps++;
                                        const auto &c = global_state::ar_profile::counters();
                                        spdlog::info(
                                            "ar_profile: decode_graph batch_size={} n_ar={} pre_barrier_ms={:.3f} "
                                            "graph_run_ms={:.3f} post_sync_ms={:.3f} "
                                            "cumulative_decode_steps={}",
                                            graph_batch_size,
                                            n_ar,
                                            pre_barrier_ms,
                                            graph_run_ms,
                                            post_sync_ms,
                                            c.decode_graph_steps);
                                        global_state::ar_profile::log_barrier_chunk_summary(
                                            "decode_step", std::nullopt, graph_batch_size);
                                    }
                                    logits = output->logits;
                                    if (general_compiler != nullptr) {
                                        general_compiler->record_graph_hit(is_prefill);
                                    }
                                } else if (general_compiler != nullptr) {
                                    general_compiler->record_graph_miss(is_prefill);
                                }
                            }
                            if (general_compiler != nullptr && is_prefill) {
                                const auto stats = general_compiler->graph_stats();
                                spdlog::debug(
                                    "[{}] prefill_graph_hit={} prefill_graph_miss={} "
                                    "piecewise_hits={} piecewise_misses={} segment_replays={} "
                                    "decode_graph_hit={} decode_graph_miss={}",
                                    info(),
                                    stats.prefill_graph_hits,
                                    stats.prefill_graph_misses,
                                    stats.piecewise_prefill_hits,
                                    stats.piecewise_prefill_misses,
                                    stats.piecewise_segment_replays,
                                    stats.decode_graph_hits,
                                    stats.decode_graph_misses);
                            }
                        }
                        // Fall back to eager mode
                        if (!logits && !piecewise_ran) {
                            global_state::hang_trace::ScopedBracket eager_bracket(
                                "eager_forward", rank_info_.tp_rank);
                            auto model_args = local_args.to_model_input(rank_info_.device);
                            // #region agent log
                            if (rank_info_.tp_rank == 0) {
                                auto &ctx = global_state::get_forward_context();
                                infinilm::agent_debug::log(
                                    "rank_worker.cpp:eager_forward",
                                    "eager_forward_begin",
                                    "A",
                                    std::string("{\"tp_size\":") + std::to_string(rank_info_.tp_size) +
                                        ",\"defer\":" +
                                        (ctx.defer_row_parallel_allreduce ? "true" : "false") +
                                        ",\"deferred_n\":" +
                                        std::to_string(ctx.deferred_allreduces.size()) + "}");
                            }
                            // #endregion
                            logits = model_->forward(model_args).logits;
                            // #region agent log
                            if (rank_info_.tp_rank == 0) {
                                auto &ctx = global_state::get_forward_context();
                                infinilm::agent_debug::log(
                                    "rank_worker.cpp:eager_forward",
                                    "eager_forward_end",
                                    "A",
                                    std::string("{\"defer\":") +
                                        (ctx.defer_row_parallel_allreduce ? "true" : "false") +
                                        ",\"deferred_n\":" +
                                        std::to_string(ctx.deferred_allreduces.size()) + "}");
                            }
                            // #endregion
                        }

                        // Random sampling (rank 0 only)
                        if (rank_info_.tp_rank == 0) {
                            const auto n_req = local_args.input_offsets.has_value()
                                                   ? local_args.input_offsets.value()->size(0) - 1
                                                   : 1;
                            const auto &final_flags = local_args.is_final_prefill_chunk;
                            std::vector<size_t> sample_rows;
                            sample_rows.reserve(n_req);
                            for (size_t i = 0; i < n_req; ++i) {
                                if (row_is_final_prefill_chunk(final_flags, i)) {
                                    sample_rows.push_back(i);
                                }
                            }

                            if (skip_sampling || sample_rows.empty()) {
                                auto output_ids{infinicore::Tensor::empty(
                                    {sample_rows.size()}, infinicore::DataType::I64, rank_info_.device)};
                                if (sample_rows.size() > 0) {
                                    set_zeros(output_ids);
                                }
                                output_ = Output{output_ids, std::nullopt};
                            } else {
                            auto temperature{local_args.temperature};
                            auto top_p{local_args.top_p};
                            auto top_k{local_args.top_k};

                            const auto &logits_shape{logits->shape()};
                            const auto &vocab_size{logits_shape[2]};
                            const auto &total_len{logits_shape[1]};
                            const auto &batch_size{logits_shape[0]};

                            int32_t *input_offsets = (int32_t *)local_args.input_offsets.value()->data();

                            auto output_ids{infinicore::Tensor::empty(
                                {sample_rows.size()}, infinicore::DataType::I64, rank_info_.device)};

                            for (size_t j = 0; j < sample_rows.size(); ++j) {
                                const size_t i = sample_rows[j];
                                auto score{logits->view({batch_size * total_len, vocab_size})->narrow({{0, size_t(input_offsets[i + 1] - 1), 1}})->view({vocab_size})};
                                auto out{output_ids->narrow({{0, j, 1}})->view({})};
                                float random_val = std::uniform_real_distribution<float>(0, 1)(rng_);
                                infinicore::op::random_sample_(
                                    out, score, random_val, top_p, top_k, temperature);
                            }

                            output_ids = output_ids->to(infinicore::Device::cpu());

                            std::optional<infinicore::Tensor> optional_logits;
                            if (local_args.return_logits) {
                                optional_logits = logits->to(infinicore::Device::cpu());
                            }

                            auto out{Output{output_ids, optional_logits}};

                            output_ = std::move(out);
                            }
                        }

                        infinicore::context::syncStream();
                        job_done_ = true;
                    }
                    cv_.notify_all();

                } catch (const std::exception &e) {
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        should_exit_ = true;
                        job_done_ = true;
                    }
                    cv_.notify_all();
                    spdlog::error("[{}] exception during forward: {}\n", info(), e.what());
                    break;
                }
            } else if (local_cmd == Command::RESET_CACHE) {
                try {
                    model_->reset_cache(local_cache_config != nullptr ? local_cache_config.get() : nullptr);
                    kv_cache_snapshot_ = global_state::get_forward_context().kv_cache_vec;
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        job_done_ = true;
                    }
                    cv_.notify_all();

                } catch (const std::exception &e) {
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        should_exit_ = true;
                        job_done_ = true;
                    }
                    cv_.notify_all();
                    spdlog::error("[{}] exception during reset_cache: {}\n", info(), e.what());
                    break;
                }
            } else if (local_cmd == Command::COMPILE) {
                try {
                    // #region agent log
                    infinilm::agent_debug::log(
                        "rank_worker.cpp:COMPILE",
                        "compile_job_begin",
                        "H3",
                        std::string("{\"tp_rank\":") + std::to_string(rank_info_.tp_rank) + "}",
                        "g3b-debug");
                    // #endregion
                    if (compiler_ != nullptr) {
                        compiler_->compile();
                    }
                    infinicore::context::syncDevice();
                    infinicore::context::flushDeferredPinnedHostFrees();
                    // #region agent log
                    infinilm::agent_debug::log(
                        "rank_worker.cpp:COMPILE",
                        "compile_job_done",
                        "H3",
                        std::string("{\"tp_rank\":") + std::to_string(rank_info_.tp_rank) + "}",
                        "g3b-debug");
                    // #endregion
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        job_done_ = true;
                    }
                    cv_.notify_all();

                } catch (const std::exception &e) {
                    // #region agent log
                    infinilm::agent_debug::log(
                        "rank_worker.cpp:COMPILE",
                        "compile_job_exception",
                        "H3",
                        std::string("{\"tp_rank\":") + std::to_string(rank_info_.tp_rank) +
                            ",\"what\":\"" + std::string(e.what()) + "\"}",
                        "g3b-debug");
                    // #endregion
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        should_exit_ = true;
                        job_done_ = true;
                    }
                    cv_.notify_all();
                    spdlog::error("[{}] exception during compile: {}\n", info(), e.what());
                    break;
                }

            } else {
                // Shouldn't reach here (no-op)
            }
        } // while
        // Some clean up should be done before exiting the thread
        if (decode_ar_event_) {
            infinicore::context::destroyEvent(decode_ar_event_);
            decode_ar_event_ = nullptr;
        }
        compiler_.reset();
    } catch (const std::exception &e) {
        // Top-level exception: ensure any waiters are woken and the thread exits cleanly.
        {
            std::lock_guard<std::mutex> lk(mutex_);
            should_exit_ = true;
            job_done_ = true;
        }
        cv_.notify_all();
        spdlog::error("[{}] fatal exception in thread_loop: {} \n", info(), e.what());
    }
}

} // namespace infinilm::engine
