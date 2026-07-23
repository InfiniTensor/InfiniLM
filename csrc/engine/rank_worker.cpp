#include "rank_worker.hpp"

#include "../global_state/ar_profile.hpp"
#include "../global_state/hang_trace.hpp"
#include "../global_state/decode_phase_profile.hpp"
#include "../global_state/global_state.hpp"
#include "../global_state/piecewise_inductor_flags.hpp"
#include "../models/model_factory.hpp"
#include "../models/models_registry.hpp"
#include "../models/qwen3/qwen3_for_causal_lm.hpp"
#include "../models/minicpm5_moe/minicpm5_moe_for_causal_lm.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/inductor_segment.hpp"
#include "infinicore/ops.hpp"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif
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

/// Derive BatchDescriptor from input shape (not scheduler phase).
/// uniform_decode: block_tables.batch == input_ids.width and every row has 1 token.
BatchDescriptor make_batch_descriptor(const RankWorker::Input &args) {
    BatchDescriptor desc;
    if (!args.block_tables.has_value() || !args.input_ids.has_value()) {
        return desc;
    }
    const size_t batch = args.block_tables.value()->size(0);
    const size_t input_width = args.input_ids.value()->size(1);
    desc.num_reqs = batch;
    desc.num_tokens = input_width;

    bool all_one_token = true;
    bool has_multi_token = false;
    if (args.input_offsets.has_value()) {
        const auto &offsets = args.input_offsets.value();
        const size_t n = offsets->size(0);
        if (n >= 2) {
            desc.num_reqs = n - 1;
            const auto *data = reinterpret_cast<const int32_t *>(offsets->data());
            desc.num_tokens = static_cast<size_t>(data[n - 1] - data[0]);
            for (size_t i = 0; i + 1 < n; ++i) {
                const int32_t row_tok = data[i + 1] - data[i];
                if (row_tok != 1) {
                    all_one_token = false;
                }
                if (row_tok > 1) {
                    has_multi_token = true;
                }
            }
        }
    } else {
        // No offsets: decode-shaped when batch == width (each req one token).
        all_one_token = (batch == input_width);
        has_multi_token = (batch != input_width);
    }

    // MIXED (decode rows + prefill rows): not uniform_decode; multi-req keeps
    // PIECEWISE off (dispatcher requires num_reqs==1 for prefill keys).
    const bool decode_shaped = (batch == input_width) && all_one_token && !has_multi_token;
    desc.uniform_decode = decode_shaped;
    if (decode_shaped) {
        desc.num_tokens = batch;
        desc.num_reqs = batch;
    }
    return desc;
}

size_t inductor_tp_rank_resolver() {
    return infinilm::global_state::get_tensor_model_parallel_rank();
}

size_t inductor_valid_seq_len_resolver() {
    return infinilm::global_state::get_forward_context().piecewise.valid_seq_len;
}

void ensure_inductor_tp_rank_resolver() {
    static std::once_flag once;
    std::call_once(once, []() {
        infinicore::op::inductor_segment_impl::set_tensor_parallel_rank_resolver(
            &inductor_tp_rank_resolver);
        infinicore::op::inductor_segment_impl::set_piecewise_valid_seq_len_resolver(
            &inductor_valid_seq_len_resolver);
    });
}

std::string piecewise_model_type(
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config = nullptr) {
    if (model_config) {
        return model_config->get_or<std::string>("model_type", "");
    }
    try {
        return infinilm::global_state::get_infinilm_config().model_config->get_or<std::string>(
            "model_type", "");
    } catch (...) {
        return "";
    }
}

void register_piecewise_pre_attn_weight_resolver(
    infinilm::InfinilmModel *model,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    if (!infinilm::global_state::piecewise_inductor_segment_enabled() || model == nullptr) {
        return;
    }
    const std::string model_type = piecewise_model_type(model_config);
    // Factory selects the concrete CausalLM from model_type. Prefer static_cast for
    // template aliases (dynamic_cast on PiecewiseTextCausalLM<> is RTTI-fragile).
    if (model_type == "minicpm5_moe") {
        auto *minicpm =
            static_cast<infinilm::models::minicpm5_moe::MiniCPM5MoeForCausalLM *>(model);
        infinicore::op::inductor_segment_impl::set_pre_attn_weight_resolver(
            [minicpm](size_t layer_idx) {
                return minicpm->model().pre_attn_external_weights(layer_idx);
            });
        spdlog::warn("piecewise: registered pre_attn weight resolver (minicpm5_moe)");
        return;
    }
    auto *qwen = dynamic_cast<infinilm::models::qwen3::Qwen3ForCausalLM *>(model);
    if (qwen != nullptr) {
        infinicore::op::inductor_segment_impl::set_pre_attn_weight_resolver(
            [qwen](size_t layer_idx) {
                return qwen->model().pre_attn_external_weights(layer_idx);
            });
        spdlog::warn("piecewise: registered pre_attn weight resolver (qwen3)");
        return;
    }
}

void register_piecewise_moe_weight_resolver(
    infinilm::InfinilmModel *model,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    if (model == nullptr) {
        return;
    }
    const std::string model_type = piecewise_model_type(model_config);
    const bool segment_on = infinilm::global_state::piecewise_inductor_segment_enabled();
    if (!segment_on) {
        if (model_type == "minicpm5_moe") {
            throw std::runtime_error(
                "piecewise: model_type=minicpm5_moe requires INFINI_PIECEWISE_INDUCTOR_SEGMENT "
                "to register MoE weight resolver (Track B)");
        }
        return;
    }
    if (model_type != "minicpm5_moe") {
        return;
    }
    // Factory guarantees MiniCPM5MoeForCausalLM when model_type matches; avoid dynamic_cast.
    auto *minicpm =
        static_cast<infinilm::models::minicpm5_moe::MiniCPM5MoeForCausalLM *>(model);
    infinicore::op::inductor_segment_impl::set_moe_weight_resolver(
        [minicpm](size_t layer_idx) {
            return minicpm->model().moe_external_weights(layer_idx);
        });
    if (!infinicore::op::inductor_segment_impl::has_moe_weight_resolver()) {
        throw std::runtime_error(
            "piecewise: set_moe_weight_resolver did not stick (has_moe_weight_resolver=false)");
    }
    spdlog::warn("piecewise: registered MoE weight resolver (minicpm5_moe)");
}

/// Pack+trim all MoE layers at PREPROCESS so first B512 request does not double-hold VRAM.
void eager_pack_minicpm5_moe_weights(
    infinilm::InfinilmModel *model,
    const std::shared_ptr<infinilm::config::ModelConfig> &model_config) {
    if (model == nullptr || piecewise_model_type(model_config) != "minicpm5_moe") {
        return;
    }
    if (!infinilm::global_state::piecewise_inductor_segment_enabled()) {
        return;
    }
    auto *minicpm =
        static_cast<infinilm::models::minicpm5_moe::MiniCPM5MoeForCausalLM *>(model);
    const size_t num_layers = minicpm->model().num_layers();
    const size_t first_k_dense =
        model_config->get_or<size_t>("first_k_dense_replace", 0);
    size_t packed = 0;
    for (size_t i = first_k_dense; i < num_layers; ++i) {
        (void)minicpm->model().moe_external_weights(i);
        ++packed;
    }
    spdlog::warn("piecewise: packed+trimmed {} MoE layers (first_k_dense_replace={})",
                 packed, first_k_dense);
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
            ensure_inductor_tp_rank_resolver();
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
                cudagraph_dispatcher_.initialize_from_env();
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
                    register_piecewise_pre_attn_weight_resolver(model_.get(), model_config_);
                    register_piecewise_moe_weight_resolver(model_.get(), model_config_);
                    eager_pack_minicpm5_moe_weights(model_.get(), model_config_);
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
                    // Re-bind MoE resolver if PREPROCESS was skipped or cleared.
                    if (!infinicore::op::inductor_segment_impl::has_moe_weight_resolver()
                        && piecewise_model_type(model_config_) == "minicpm5_moe") {
                        register_piecewise_moe_weight_resolver(model_.get(), model_config_);
                    }
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
                        // scheduling_mode is logging / Python row updates only;
                        // CG path uses BatchDescriptor dispatch below.
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
                        const BatchDescriptor batch_desc = make_batch_descriptor(local_args);
                        const auto [cg_mode, cg_key] = cudagraph_dispatcher_.dispatch(batch_desc);
                        // Align phase flags with dispatch when scheduler mode unset.
                        if (!local_args.scheduling_mode.has_value()) {
                            if (cg_mode == CudaGraphRuntimeMode::Piecewise) {
                                is_prefill = true;
                                is_mixed = false;
                            } else if (cg_mode == CudaGraphRuntimeMode::Full) {
                                is_prefill = false;
                                is_mixed = false;
                            } else if (batch_desc.uniform_decode) {
                                is_prefill = false;
                            } else if (batch_desc.num_reqs > 1 && !batch_desc.uniform_decode) {
                                // Ragged / mixed → eager (NONE).
                                is_mixed = true;
                            }
                        }
                        if (global_state::hang_trace::enabled() && rank_info_.tp_rank == 0) {
                            spdlog::info(
                                "hang_trace: run_job_begin mode={} cg_mode={} n_req={} "
                                "batch_size={} num_tokens={} uniform_decode={} is_mixed={}",
                                mode_str,
                                cudagraph_runtime_mode_cstr(cg_mode),
                                n_req,
                                batch_size,
                                batch_desc.num_tokens,
                                batch_desc.uniform_decode,
                                is_mixed);
                        }
                        auto model_input = local_args.to_model_input(infinicore::Device::cpu());
                        if (compiler_ != nullptr && cg_mode != CudaGraphRuntimeMode::None) {
                            auto *general_compiler = dynamic_cast<GeneralCompiler *>(compiler_.get());
                            if (cg_mode == CudaGraphRuntimeMode::Piecewise && general_compiler != nullptr
                                && general_compiler->native_piecewise_enabled()) {
                                // run_prefill = piecewise backend replay (not scheduler phase).
                                const bool profile = rank_worker_profile_enabled();
                                const double t_pw0 = profile ? monotonic_ms() : 0.0;
                                global_state::hang_trace::ScopedBracket pw_bracket(
                                    "native_piecewise", rank_info_.tp_rank);
                                if (profile) {
                                    spdlog::info(
                                        "rank_worker_profile: RUN native_piecewise begin "
                                        "cg_mode=PIECEWISE n_req={} num_tokens={} key_tokens={}",
                                        batch_desc.num_reqs,
                                        batch_desc.num_tokens,
                                        cg_key.num_tokens);
                                }
                                if (auto pw_logits = general_compiler->run_native_piecewise_prefill(model_input)) {
                                    logits = *pw_logits;
                                    piecewise_ran = true;
                                } else if (general_compiler->native_piecewise_last_prefill_executed()) {
                                    piecewise_ran = true;
                                    if (!any_final_prefill_chunk(model_input.is_final_prefill_chunk)) {
                                        skip_sampling = true;
                                    }
                                }
                                if (profile) {
                                    spdlog::info(
                                        "rank_worker_profile: RUN native_piecewise end piecewise_ran={} "
                                        "skip_sampling={} wall_ms={:.3f}",
                                        piecewise_ran,
                                        skip_sampling,
                                        monotonic_ms() - t_pw0);
                                }
                            } else if (cg_mode == CudaGraphRuntimeMode::Full && general_compiler != nullptr
                                       && general_compiler->native_piecewise_decode_enabled()) {
                                const bool profile = rank_worker_profile_enabled();
                                const bool decode_prof =
                                    global_state::decode_phase_profile::enabled();
                                if (decode_prof) {
                                    global_state::decode_phase_profile::reset();
                                    global_state::decode_phase_profile::decode_step_active() = true;
                                }
                                const double t_fwd0 =
                                    decode_prof ? global_state::decode_phase_profile::monotonic_ms()
                                                : 0.0;
                                const double t_pw0 = profile ? monotonic_ms() : 0.0;
                                global_state::hang_trace::ScopedBracket pw_bracket(
                                    "native_piecewise_decode", rank_info_.tp_rank);
                                try {
                                    if (auto pw_logits =
                                            general_compiler->run_native_piecewise_decode(model_input)) {
                                        logits = *pw_logits;
                                        piecewise_ran = true;
                                        if (auto *gc = dynamic_cast<GeneralCompiler *>(compiler_.get())) {
                                            gc->record_graph_hit(/*is_prefill=*/false);
                                        }
                                    }
                                } catch (...) {
                                    if (decode_prof) {
                                        global_state::decode_phase_profile::decode_step_active() =
                                            false;
                                    }
                                    throw;
                                }
                                if (decode_prof) {
                                    global_state::decode_phase_profile::counters().eager_forward_ms +=
                                        global_state::decode_phase_profile::monotonic_ms() - t_fwd0;
                                    global_state::decode_phase_profile::decode_step_active() = false;
                                }
                                if (profile) {
                                    const auto stats = general_compiler->graph_stats();
                                    spdlog::info(
                                        "rank_worker_profile: RUN native_piecewise_decode "
                                        "piecewise_ran={} wall_ms={:.3f} decode_hits={} "
                                        "decode_misses={} device_segs={} segment_replays={}",
                                        piecewise_ran,
                                        monotonic_ms() - t_pw0,
                                        stats.piecewise_decode_hits,
                                        stats.piecewise_decode_misses,
                                        stats.piecewise_decode_device_segments,
                                        stats.piecewise_segment_replays);
                                }
                            }
                            if (cg_mode == CudaGraphRuntimeMode::Full && !logits && !piecewise_ran) {
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
                                        // FULL decode replay: FA host-break; MoE HB under
                                        // full_and_piecewise unless INFINI_MOE_TRITON_CAPTURE=1.
                                        infinicore::context::InferencePhaseGuard phase_guard(
                                            infinicore::context::InferencePhase::Decode);
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
                                        general_compiler->record_graph_hit(/*is_prefill=*/false);
                                    }
                                } else if (general_compiler != nullptr) {
                                    general_compiler->record_graph_miss(/*is_prefill=*/false);
                                }
                            }
                            if (general_compiler != nullptr
                                && cg_mode == CudaGraphRuntimeMode::Piecewise) {
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
                        // Fall back to eager mode (NONE, or PIECEWISE/FULL miss)
                        if (!logits && !piecewise_ran) {
                            global_state::hang_trace::ScopedBracket eager_bracket(
                                "eager_forward", rank_info_.tp_rank);
                            auto model_args = local_args.to_model_input(rank_info_.device);
                            const bool decode_prof =
                                !is_prefill && global_state::decode_phase_profile::enabled();
                            if (decode_prof) {
                                global_state::decode_phase_profile::reset();
                                global_state::decode_phase_profile::decode_step_active() = true;
                            }
                            const double t_fwd0 =
                                decode_prof ? global_state::decode_phase_profile::monotonic_ms() : 0.0;
                            try {
                                logits = model_->forward(model_args).logits;
                            } catch (...) {
                                if (decode_prof) {
                                    global_state::decode_phase_profile::decode_step_active() = false;
                                }
                                throw;
                            }
                            if (decode_prof) {
                                global_state::decode_phase_profile::counters().eager_forward_ms +=
                                    global_state::decode_phase_profile::monotonic_ms() - t_fwd0;
                                global_state::decode_phase_profile::decode_step_active() = false;
                            }
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

                            const bool decode_prof =
                                !is_prefill && global_state::decode_phase_profile::enabled();
                            const double t_sample0 =
                                decode_prof ? global_state::decode_phase_profile::monotonic_ms() : 0.0;

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
                            if (decode_prof) {
                                global_state::decode_phase_profile::counters().sample_ms +=
                                    global_state::decode_phase_profile::monotonic_ms() - t_sample0;
                            }
                        }

                        {
                            const bool decode_prof =
                                !is_prefill && global_state::decode_phase_profile::enabled();
                            const double t_sync0 =
                                decode_prof ? global_state::decode_phase_profile::monotonic_ms() : 0.0;
                            // TP=1 + LIGHT_SYNC: sample D2H already drains the stream.
                            // Skip a redundant full syncStream (FA correctness does not need it).
                            const bool skip_end_sync = decode_light_sync_enabled()
                                                       && rank_info_.tp_size == 1
                                                       && !skip_sampling
                                                       && rank_info_.tp_rank == 0;
                            if (!skip_end_sync) {
                                infinicore::context::syncStream();
                            }
                            if (decode_prof) {
                                global_state::decode_phase_profile::counters().sync_ms +=
                                    global_state::decode_phase_profile::monotonic_ms() - t_sync0;
                                const auto n_req = local_args.input_offsets.has_value()
                                                       ? local_args.input_offsets.value()->size(0) - 1
                                                       : 1;
                                global_state::decode_phase_profile::log_step_if_due(n_req);
                            }
                        }
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
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
                    c10::cuda::CUDAGuard device_guard(static_cast<int>(rank_info_.device.getIndex()));
#endif
                    if (compiler_ != nullptr) {
                        compiler_->compile();
                    }
                    infinicore::context::syncDevice();
                    infinicore::context::flushDeferredPinnedHostFrees();
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
