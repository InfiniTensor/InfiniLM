#include "rank_worker.hpp"
#include "../engine/distributed/communication_group.hpp"
#include "../layers/moe/ep/ep_config.hpp"
#include "../models/model_factory.hpp"
#include "../weights/weights.hpp"
#include "compiler/paged_compiler.hpp"
#include "compiler/paged_decode_bucket.hpp"
#include "compiler/static_batching_compiler.hpp"
#include "nsa_run_failure.hpp"
#include "request_sampling.hpp"
#include "runtime/runtime.hpp"
#include <algorithm>
#include <cstring>
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::engine {

namespace {
infinilm::runtime::Tensor clone_nsa_cpu(const std::optional<infinilm::runtime::Tensor> &value,
                                        infinilm::runtime::DataType dtype, std::size_t rank) {
    if (!value || !value.value() || value.value()->device().type() != infinilm::runtime::Device::Type::kCpu
        || value.value()->dtype() != dtype || value.value()->ndim() != rank || !value.value()->is_contiguous()) {
        throw std::invalid_argument("NSA requires contiguous original CPU input metadata");
    }
    auto storage = std::make_shared<std::vector<std::byte>>(value.value()->nbytes());
    auto copy = infinilm::runtime::Tensor::from_blob(storage->data(), value.value()->shape(), dtype,
                                                     infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu}, storage);
    std::memcpy(copy->data(), value.value()->data(), value.value()->nbytes());
    return copy;
}
template <class T>
std::vector<T> nsa_values(const std::optional<infinilm::runtime::Tensor> &t) {
    const auto *p = reinterpret_cast<const T *>(t.value()->data());
    return {p, p + t.value()->numel()};
}
nsa::Upload nsa_upload(const RankWorker::Input &a) {
    return {nsa_values<std::int32_t>(a.past_sequence_lengths),
            nsa_values<std::int32_t>(a.total_sequence_lengths),
            nsa_values<std::int32_t>(a.input_offsets), nsa_values<std::int32_t>(a.cu_seqlens),
            nsa_values<std::int32_t>(a.block_tables), a.block_tables.value()->size(1),
            nsa_values<std::int64_t>(a.slot_mapping)};
}
void freeze_nsa_cpu(RankWorker::Input &a) {
    using D = infinilm::runtime::DataType;
    a.past_sequence_lengths = clone_nsa_cpu(a.past_sequence_lengths, D::kInt32, 1);
    a.total_sequence_lengths = clone_nsa_cpu(a.total_sequence_lengths, D::kInt32, 1);
    a.input_offsets = clone_nsa_cpu(a.input_offsets, D::kInt32, 1);
    a.cu_seqlens = clone_nsa_cpu(a.cu_seqlens, D::kInt32, 1);
    a.block_tables = clone_nsa_cpu(a.block_tables, D::kInt32, 2);
    a.slot_mapping = clone_nsa_cpu(a.slot_mapping, D::kInt64, 1);
    a.input_ids = clone_nsa_cpu(a.input_ids, D::kInt64, 2);
    // Text MRoPE is the actual processor's [3, tokens] CPU tensor.
    a.position_ids = clone_nsa_cpu(a.position_ids, D::kInt64, 2);
}
} // namespace

RankWorker::RankWorker(
    std::shared_ptr<infinilm::global_state::InfinilmConfig> infinilm_config,
    const distributed::RankInfo &rank_info,
    const cache::CacheConfig *cache_config,
    RankBarrier *barrier,
    bool enable_graph_compiling,
    backends::AttentionBackend attention_backend,
    std::uint64_t nsa_pool_epoch)
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
    nsa_pool_epoch_ = nsa_pool_epoch;
    // start the thread
    thread_ = std::thread(&RankWorker::thread_loop, this);
}

void RankWorker::refresh_finite_paged_decode_profile() {
    finite_paged_decode_profile_.reset();
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(
        model_->get_cache_config());
    if (paged_config == nullptr) {
        return;
    }
    const auto profile = make_finite_paged_decode_profile(
        *paged_config, model_config_.get(), model_->supports_kv_only_prepare(),
        rank_info_.device.type(),
        PagedGraphConfig{attention_backend_, rank_info_.tp_size, rank_info_.tp_rank, rank_info_.pp_size});
    if (supports_finite_paged_decode_bucket(profile)) {
        finite_paged_decode_profile_ = profile;
    }
}

void RankWorker::wait_for_init() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });
    if (should_exit_) {
        auto failure = failure_;
        lk.unlock();
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw std::runtime_error("RankWorker failed to initialize");
    }
}

RankWorker::~RankWorker() {
    close();
    // Fail-stop policy: a failed drain must not destruct backing/Runtime/communicators.
    if (nsa_terminal_undrained_) {
        std::terminate();
    }
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

//------------------------------------------------------
// load_param -- synchronous (blocks until worker finishes loading)
//------------------------------------------------------
void RankWorker::load_param(const std::string &name,
                            const infinilm::runtime::Tensor &param) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // If the worker is stopping, don't submit new jobs.
        if (should_exit_) {
            if (failure_) {
                std::rethrow_exception(failure_);
            }
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
        auto failure = failure_;
        lk.unlock();
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw std::runtime_error("RankWorker stopped while loading parameter");
    }
}

//------------------------------------------------------
// load_params -- synchronous batch load
//------------------------------------------------------
void RankWorker::load_params(const std::unordered_map<std::string, infinilm::runtime::Tensor> &params, bool strict) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (should_exit_) {
            if (failure_) {
                std::rethrow_exception(failure_);
            }
            throw std::runtime_error("RankWorker is closing; cannot load_params");
        }

        pending_params_ = params;
        pending_params_strict_ = strict;
        job_cmd_ = Command::LOAD_BATCH;
        has_job_ = true;
        job_done_ = false;
    }
    cv_.notify_all();

    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return job_done_ || should_exit_; });

    if (should_exit_) {
        auto failure = failure_;
        lk.unlock();
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw std::runtime_error("RankWorker stopped while loading parameters");
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
            if (failure_) {
                std::rethrow_exception(failure_);
            }
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
        auto failure = failure_;
        lk.unlock();
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw std::runtime_error("RankWorker stopped while processing weights");
    }
}

void RankWorker::validate_w4_weights() { submit_w4_phase(Command::VALIDATE_W4); }
void RankWorker::prepare_w4_weights() { submit_w4_phase(Command::PREPARE_W4); }
void RankWorker::commit_w4_weights() { submit_w4_phase(Command::COMMIT_W4); }

void RankWorker::submit_w4_phase(Command command) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        // If the worker is stopping, don't submit new jobs.
        if (should_exit_) {
            if (failure_) {
                std::rethrow_exception(failure_);
            }
            throw std::runtime_error("RankWorker is closing; cannot process_weights_after_loading");
        }

        job_cmd_ = command;
        has_job_ = true;
        job_done_ = false;
    }
    cv_.notify_all();

    // Wait for job completion
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return job_done_ || should_exit_; });

    if (should_exit_) {
        auto failure = failure_;
        lk.unlock();
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw std::runtime_error("RankWorker stopped while processing weights");
    }
}

//------------------------------------------------------
// state_dict --
//------------------------------------------------------
std::unordered_map<std::string, infinilm::weights::Weight> RankWorker::state_dict() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });

    if (failure_) {
        std::rethrow_exception(failure_);
    }
    if (!model_) {
        throw std::runtime_error("state_dict called before model initialization");
    }

    return model_->state_dict();
}

std::vector<std::string> RankWorker::state_dict_keys() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });

    if (failure_) {
        std::rethrow_exception(failure_);
    }
    if (!model_) {
        throw std::runtime_error("state_dict_keys called before model initialization");
    }

    return model_->state_dict_keys();
}

std::vector<std::string> RankWorker::state_required_keys() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });
    if (failure_) {
        std::rethrow_exception(failure_);
    }
    if (!model_) {
        throw std::runtime_error("state_required_keys called before model initialization");
    }
    return model_->state_required_keys();
}

//------------------------------------------------------
// run -- asynchronous
//------------------------------------------------------
void RankWorker::run(const Input &args) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (should_exit_) {
        if (failure_) {
            std::rethrow_exception(failure_);
        }
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
        if (failure_) {
            std::rethrow_exception(failure_);
        }
        throw std::runtime_error("RankWorker is closing; cannot run");
    }

    if (nsa_pool_epoch_) {
        throw std::invalid_argument("NSA owner slice excludes COMPILE before command mutation");
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
        auto failure = failure_;
        lk.unlock();
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw std::runtime_error("RankWorker stopped during run");
    }
}

void RankWorker::reset_cache(const cache::CacheConfig *new_config, std::uint64_t nsa_pool_epoch) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (should_exit_) {
        if (failure_) {
            std::rethrow_exception(failure_);
        }
        throw std::runtime_error("RankWorker is closing; cannot reset_cache");
    }

    if (nsa_pool_epoch_ && (!nsa_pool_epoch || nsa_pool_epoch == nsa_pool_epoch_)) {
        throw std::invalid_argument("NSA reset requires a new native pool epoch");
    }
    pending_nsa_pool_epoch_ = nsa_pool_epoch;
    // Store both the position and the new config
    pending_cache_config_ = new_config->unique_copy();
    job_cmd_ = Command::RESET_CACHE;
    has_job_ = true;
    job_done_ = false;
    cv_.notify_all();
}

//------------------------------------------------------
// get kv cache
//------------------------------------------------------
std::uint64_t RankWorker::get_nsa_pool_epoch() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });
    if (should_exit_ || !nsa_pool_epoch_) {
        if (failure_) {
            std::rethrow_exception(failure_);
        }
        throw std::logic_error("NSA pool query requires a live worker");
    }
    return nsa_pool_epoch_;
}

bool RankWorker::drain_nsa_owner_noexcept() noexcept {
    if (!nsa_pool_epoch_ || !nsa_drain_required_) {
        return true;
    }
    if (nsa_terminal_undrained_) {
        return false;
    }
    try {
        if (runtime_ == nullptr) {
            throw std::logic_error("NSA owner Runtime absent during drain");
        }
        runtime_->syncStream();
        nsa_drain_required_ = false;
        return true;
    } catch (...) {
        nsa_terminal_undrained_ = true; // sticky; model/plans/context remain strongly held
        try {
            spdlog::critical("NSA owner drain failed; backing retained; shutdown is fail-stop");
        } catch (...) {}
        return false;
    }
}

std::vector<infinilm::runtime::Tensor> RankWorker::get_kv_cache() {
    std::unique_lock<std::mutex> lk(mutex_);
    if (nsa_pool_epoch_) {
        throw std::invalid_argument("NSA borrowed KV writer unsupported");
    }
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });

    if (should_exit_) {
        auto failure = failure_;
        lk.unlock();
        if (failure) {
            std::rethrow_exception(failure);
        }
        throw std::runtime_error("RankWorker stopped; cannot get_cache_vec");
    }

    ASSERT(forward_context_.kv_cache_vec.size() > 0 && "RankWorker::get_kv_cache(): kv_cache_vec is empty");

    return forward_context_.kv_cache_vec;
}

//------------------------------------------------------
// close -- request shutdown and join thread
//------------------------------------------------------
void RankWorker::close() {
    request_stop();
    join();
}

void RankWorker::request_stop() noexcept {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        should_exit_ = true;
        has_job_ = false; // don't keep old jobs pending
        job_cmd_ = Command::STOP;
    }
    cv_.notify_all();
}

void RankWorker::join() noexcept {
    if (thread_.joinable()) {
        thread_.join();
    }
    // Also covers InferEngine constructor failure before its destructor exists.
    if (nsa_terminal_undrained_) {
        std::terminate();
    }
}

void RankWorker::fail(const char *operation, std::exception_ptr error) noexcept {
    if (!error) {
        error = std::make_exception_ptr(std::runtime_error("unknown rank worker failure"));
    }
    try {
        if (runtime_ && model_config_->get<std::string>("model_type") == "kimi_k3") {
            runtime_->syncStream();
        }
    } catch (...) {
        std::terminate();
    }
    if (barrier_ != nullptr) {
        barrier_->abort(error);
    }
    {
        std::lock_guard<std::mutex> lk(mutex_);
        if (!failure_) {
            failure_ = error;
        }
        init_done_ = true;
        should_exit_ = true;
        job_done_ = true;
    }
    cv_.notify_all();

    try {
        std::rethrow_exception(error);
    } catch (const std::exception &e) {
        try {
            spdlog::error("[{}] exception during {}: {}", rank_info_.to_string(), operation, e.what());
        } catch (...) {
        }
    } catch (...) {
        try {
            spdlog::error("[{}] unknown exception during {}", rank_info_.to_string(), operation);
        } catch (...) {
        }
    }
}

//------------------------------------------------------
// get_output (thread safe)
//------------------------------------------------------
RankWorker::Output RankWorker::get_output() {
    std::lock_guard<std::mutex> lock(mutex_);
    return output_;
}

//------------------------------------------------------
// thread_loop
//------------------------------------------------------
void RankWorker::thread_loop() {
    try {
        {
            std::lock_guard<std::mutex> lk(mutex_);

            // Initialize device & model outside of holding the main mutex to avoid blocking callers.
            runtime_ = infinilm::runtime::Runtime::create(rank_info_.device);
            runtime_->activate();
            forward_context_.runtime = runtime_.get();
            // Submission failure is terminal for this dedicated job-host process.
            // This also opts every worker into the existing strict failure-drain paths.
            runtime_->enableOwnerDrainFailStop();
            nsa_drain_required_ = nsa_pool_epoch_ != 0;

            // Create model using factory (may be expensive)
            // Construction EP configuration uses this worker's owned values.
            const auto construction_ep = layers::moe::make_ep_config(infinilm_config_->moe_ep_backend,
                                                                     infinilm_config_->moe_ep_size,
                                                                     rank_info_.tp_size, rank_info_.tp_rank);
            model_ = InfinilmModelFactory::createModel(
                model_config_, *runtime_, rank_info_, attention_backend_, construction_ep,
                pending_cache_config_ != nullptr ? pending_cache_config_.get() : nullptr,
                forward_context_, infinilm_config_->use_mla);
            if (model_config_->get<std::string>("model_type") == "kimi_k3") {
                runtime_->syncStream();
            }
            refresh_finite_paged_decode_profile();
            if (enable_graph_compiling_ && rank_info_.pp_size == 1) {
                PagedGraphConfig{attention_backend_, rank_info_.tp_size, rank_info_.tp_rank, rank_info_.pp_size}.validate();
            }

            init_done_ = true;
        }
        cv_.notify_all();

        // Main loop: wait for jobs or exit
        while (true) {
            Command local_cmd = Command::INIT;
            std::string local_param_name;
            infinilm::runtime::Tensor local_param;
            std::unordered_map<std::string, infinilm::runtime::Tensor> local_params;
            bool local_params_strict = true;
            Input local_args;
            std::unique_ptr<cache::CacheConfig> local_cache_config;
            std::uint64_t local_nsa_pool_epoch = 0;

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
                    local_param = std::move(pending_param_);
                } else if (local_cmd == Command::LOAD_BATCH) {
                    local_params = std::move(pending_params_);
                    // strict is copied with the batch because loading runs on
                    // the worker thread after the caller releases the mutex.
                    local_params_strict = pending_params_strict_;
                    pending_params_strict_ = true;
                    pending_params_.clear();
                } else if (local_cmd == Command::PREPROCESS) {

                } else if (local_cmd == Command::RUN) {
                    local_args = pending_args_;
                } else if (local_cmd == Command::RESET_CACHE) {
                    // Epoch follows the actual serialized reset command payload.
                    local_nsa_pool_epoch = pending_nsa_pool_epoch_;
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
                    if (nsa_pool_epoch_) {
                        nsa_drain_required_ = true;
                    }
                    model_->load_parameter(*runtime_, local_param_name, local_param);
                    local_param.reset();
                } catch (...) {
                    fail("load_parameter", std::current_exception());
                    break;
                }

                // signal completion
                {
                    std::lock_guard<std::mutex> lk(mutex_);
                    job_done_ = true;
                }
                cv_.notify_all();

            } else if (local_cmd == Command::LOAD_BATCH) {
                try {
                    if (nsa_pool_epoch_) {
                        nsa_drain_required_ = true;
                    }
                    model_->load_parameters_no_sync(*runtime_, local_params, local_params_strict);
                    runtime_->syncStream();
                    // Completion includes releasing the source owners, not only H2D.
                    local_params.clear();
                } catch (...) {
                    fail("load_parameters", std::current_exception());
                    break;
                }

                {
                    std::lock_guard<std::mutex> lk(mutex_);
                    job_done_ = true;
                }
                cv_.notify_all();

            } else if (local_cmd == Command::VALIDATE_W4 || local_cmd == Command::PREPARE_W4 || local_cmd == Command::COMMIT_W4) {
                try {
                    runtime_->assertOwnerThread();
                    if (runtime_->graphCaptureState() != infinilm::execution::graph::GraphManager::CaptureState::kInactive) {
                        throw std::logic_error("W4 finalization requires inactive graph state");
                    }
                    if (local_cmd == Command::VALIDATE_W4) {
                        if (w4_plan_) {
                            throw std::logic_error("W4 validation is single-use");
                        }
                        w4_plan_ = model_->validate_w4_weights(
                            *runtime_, infinilm_config_->pre_transpose, rank_info_,
                            infinilm_config_->moe_ep_backend, infinilm_config_->moe_ep_size);
                    } else if (local_cmd == Command::PREPARE_W4) {
                        if (!w4_plan_) {
                            throw std::logic_error("W4 prepare lacks validated exact coverage");
                        }
                        w4_plan_->prepare(*runtime_);
                        runtime_->syncStream();
                        runtime_->trimMemory();
                        w4_plan_->seal();
                    } else {
                        if (!w4_plan_) {
                            throw std::logic_error("W4 commit lacks prepared plan");
                        }
                        w4_plan_->commit();
                        w4_plan_.reset();
                    }
                } catch (...) {
                    const auto error = std::current_exception();
                    // Staged strong owners/candidates remain held until drain.
                    try {
                        runtime_->syncStream();
                    } catch (...) { std::terminate(); }
                    fail("W4 finalization", error);
                    break;
                }
                {
                    std::lock_guard<std::mutex> lk(mutex_);
                    job_done_ = true;
                }
                cv_.notify_all();
            } else if (local_cmd == Command::PREPROCESS) {
                // Handle preprocess command
                try {
                    if (nsa_pool_epoch_) {
                        nsa_drain_required_ = true;
                    }
                    model_->process_weights_after_loading(*runtime_, infinilm_config_->pre_transpose);
                    runtime_->syncStream();
                    runtime_->trimMemory();
                } catch (...) {
                    fail("process_weights_after_loading", std::current_exception());
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
                    if (nsa_pool_epoch_) {
                        nsa_drain_required_ = true;
                    }
                    {
                        std::lock_guard<std::mutex> lk(mutex_);

                        infinilm::runtime::Tensor logits;
                        infinilm::runtime::Tensor hidden_states;
                        std::optional<infinilm::execution::graph::Graph::ExecutionReport> graph_execution;
                        auto effective_args = local_args;
                        std::shared_ptr<const nsa::Proof> nsa_proof;
                        // These retained RUN locals are OUTSIDE the inner try.
                        // Its catch sees the real exception before these locals unwind.
                        try {
                            if (nsa_pool_epoch_) {
                                if (runtime_->isGraphRecording() || rank_info_.pp_size != 1
                                    || local_args.sample_all_positions || local_args.target_hidden_states || local_args.pixel_values
                                    || local_args.image_bound || local_args.tgt_sizes || local_args.image_grid_thw
                                    || local_args.image_req_ids || local_args.visual_token_ranges
                                    || local_args.mamba_init_state_indices || local_args.mamba_final_state_indices
                                    || !local_args.nsa_owner_snapshot || local_args.nsa_owner_snapshot->pool_epoch != nsa_pool_epoch_) {
                                    throw std::invalid_argument("NSA RUN requires current ordinary text owner DTO and eager worker");
                                }
                                freeze_nsa_cpu(effective_args);
                                const auto original = nsa_upload(effective_args);
                                if (!(original == local_args.nsa_owner_snapshot->upload)) {
                                    throw std::invalid_argument("NSA submitted CPU metadata changed after producer freeze");
                                }
                                nsa::validate(*local_args.nsa_owner_snapshot, original);
                                if (effective_args.input_ids.value()->size(0) != 1
                                    || effective_args.input_ids.value()->size(1) != original.slots.size()
                                    || effective_args.position_ids.value()->size(0) != 3
                                    || effective_args.position_ids.value()->size(1) != original.slots.size()) {
                                    throw std::invalid_argument("NSA packed input/position shape mismatch");
                                }
                            }
                            size_t logical_kv_upper_bound = 0;
                            if (finite_paged_decode_profile_
                                && !local_args.sample_all_positions
                                && effective_args.input_ids && *effective_args.input_ids
                                && effective_args.input_ids.value()->ndim() == 2
                                && effective_args.input_ids.value()->size(0) == 1
                                && effective_args.total_sequence_lengths && *effective_args.total_sequence_lengths
                                && effective_args.input_offsets && *effective_args.input_offsets
                                && effective_args.block_tables && *effective_args.block_tables) {
                                const auto &lengths = effective_args.total_sequence_lengths.value();
                                const auto &offsets = effective_args.input_offsets.value();
                                const auto &blocks = effective_args.block_tables.value();
                                const size_t batch = lengths->ndim() == 1 ? lengths->size(0) : 0;
                                if (offsets->device().type() != infinilm::runtime::Device::Type::kCpu
                                    || offsets->dtype() != infinilm::runtime::DataType::kInt32
                                    || !offsets->is_contiguous() || offsets->ndim() != 1
                                    || offsets->size(0) != batch + 1) {
                                    throw std::invalid_argument("paged decode requires CPU int32 contiguous input offsets");
                                }
                                const auto *offset_data = reinterpret_cast<const int32_t *>(offsets->data());
                                if (is_packed_paged_decode(
                                        effective_args.input_ids.value()->size(0),
                                        effective_args.input_ids.value()->size(1),
                                        offset_data, batch + 1,
                                        batch)) {
                                    if (lengths->device().type() != infinilm::runtime::Device::Type::kCpu
                                        || lengths->dtype() != infinilm::runtime::DataType::kInt32
                                        || !lengths->is_contiguous() || lengths->ndim() != 1
                                        || lengths->size(0) != batch
                                        || blocks->device().type() != infinilm::runtime::Device::Type::kCpu
                                        || blocks->dtype() != infinilm::runtime::DataType::kInt32
                                        || !blocks->is_contiguous() || blocks->ndim() != 2
                                        || blocks->size(0) != batch) {
                                        throw std::invalid_argument("paged decode bucket requires CPU int32 contiguous lengths and table");
                                    }
                                    const size_t source_width = blocks->size(1);
                                    if (source_width == 0) {
                                        throw std::invalid_argument("paged decode bucket requires nonempty table");
                                    }
                                    const auto *length_data = reinterpret_cast<const int32_t *>(lengths->data());
                                    const auto *block_data = reinterpret_cast<const int32_t *>(blocks->data());
                                    auto choice = choose_paged_decode_bucket(
                                        length_data, batch, block_data, blocks->numel(), source_width,
                                        finite_paged_decode_profile_->block_size,
                                        finite_paged_decode_profile_->num_blocks);
                                    auto storage = std::make_shared<std::vector<int32_t>>(std::move(choice.block_table));
                                    auto normalized = infinilm::runtime::Tensor::from_blob(storage->data(),
                                                                                           {choice.batch_size, choice.table_width}, infinilm::runtime::DataType::kInt32,
                                                                                           infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu}, storage);
                                    effective_args.block_tables = std::move(normalized);
                                    logical_kv_upper_bound = choice.logical_kv_upper_bound;
                                }
                            }
                            if (nsa_pool_epoch_) {
                                auto proof = std::make_shared<nsa::Proof>();
                                proof->owner = *local_args.nsa_owner_snapshot;
                                proof->owner.upload = nsa_upload(effective_args); // actual normalized CPU tensor
                                proof->one_tail = nsa::validate(proof->owner, proof->owner.upload);
                                if (nsa_run_serial_ == std::numeric_limits<std::uint64_t>::max()) {
                                    throw std::overflow_error("NSA invocation exhausted");
                                }
                                proof->runtime = runtime_.get();
                                proof->worker = static_cast<std::uint64_t>(rank_info_.tp_rank) + 1;
                                proof->invocation = ++nsa_run_serial_;
                                nsa_proof = proof;
                                forward_context_.nsa_owner = proof;
                                model_->prepare_paged_owner(forward_context_); // all layers BEFORE any input/device/raw write
                            }
                            const char *eager_reason = !enable_graph_compiling_        ? "graphs_disabled"
                                                     : rank_info_.pp_size != 1         ? "pipeline_graph_not_supported"
                                                     : local_args.sample_all_positions ? "all_positions_not_compiled"
                                                                                       : "no_matching_plan";
                            // Packed all-position runs do not match the one-token-per-request
                            // output shape captured by the compiled decode graphs.
                            if (!local_args.sample_all_positions && enable_graph_compiling_ && rank_info_.pp_size == 1) {
                                GraphCompiler::Compiled compiled{nullptr, nullptr};
                                {
                                    const auto lookup_input = effective_args.to_model_input(
                                        infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu},
                                        forward_context_,
                                        false,
                                        true,
                                        logical_kv_upper_bound, nsa_proof);
                                    if (compiler_) {
                                        compiled = compiler_->get_compiled(lookup_input);
                                    }
                                }
                                auto [graph, output] = std::move(compiled);
                                if (graph != nullptr && output != nullptr) {
                                    graph->run();
                                    graph_execution = graph->execution_report();
                                    logits = output->logits;
                                    hidden_states = output->hidden_states;
                                }
                            }
                            // Fall back to eager mode
                            if (!logits) {
                                auto model_args = effective_args.to_model_input(
                                    rank_info_.device,
                                    forward_context_,
                                    attention_backend_ == backends::AttentionBackend::STATIC_ATTN,
                                    false,
                                    logical_kv_upper_bound, nsa_proof);
                                if (attention_backend_ == backends::AttentionBackend::STATIC_ATTN
                                    && model_config_->get_kv_quant_scheme() == quantization::KVQuantAlgo::NONE
                                    && model_args.input_ids && model_args.input_ids.value()->ndim() == 2
                                    && model_args.input_ids.value()->size(1) == 1) {
                                    const auto cache = std::find_if(
                                        forward_context_.kv_cache_vec.begin(), forward_context_.kv_cache_vec.end(),
                                        [](const infinilm::runtime::Tensor &t) { return static_cast<bool>(t); });
                                    if (cache != forward_context_.kv_cache_vec.end() && (*cache)->ndim() == 5) {
                                        const auto batch = model_args.input_ids.value()->size(0);
                                        const auto capacity = (*cache)->size(1);
                                        if (batch == 0 || batch > capacity
                                            || !effective_args.past_sequence_lengths
                                            || !effective_args.total_sequence_lengths) {
                                            throw std::invalid_argument("Static decode requires one valid cache row and length pair per request");
                                        }
                                        for (const auto &layer_cache : forward_context_.kv_cache_vec) {
                                            if (layer_cache && (layer_cache->ndim() != 5 || layer_cache->size(1) != capacity || layer_cache->size(3) != (*cache)->size(3))) {
                                                throw std::invalid_argument("Static decode requires matching cache row layouts");
                                            }
                                        }
                                        static_decode_metadata_.prepare(
                                            *runtime_, batch, capacity, (*cache)->size(3));
                                        auto description = static_decode_metadata_.update(
                                            *runtime_, *effective_args.past_sequence_lengths,
                                            *effective_args.total_sequence_lengths,
                                            effective_args.block_tables, local_args.slot_mapping);
                                        model_args.block_tables = static_decode_metadata_.block_tables();
                                        model_args.slot_mapping = static_decode_metadata_.slot_mapping();
                                        forward_context_.attn_metadata.block_tables = model_args.block_tables;
                                        forward_context_.attn_metadata.slot_mapping = model_args.slot_mapping;
                                        forward_context_.attn_metadata.static_decode_host = description;
                                    }
                                }
                                if (attention_backend_ == backends::AttentionBackend::STATIC_ATTN
                                    && model_->supports_kv_only_prepare()
                                    && model_config_->get_kv_quant_scheme() == quantization::KVQuantAlgo::NONE
                                    && model_args.input_ids && model_args.input_ids.value()->ndim() == 2
                                    && model_args.input_ids.value()->size(1) > 1) {
                                    const auto batch = model_args.input_ids.value()->size(0);
                                    const auto query_length = model_args.input_ids.value()->size(1);
                                    const auto cache = std::find_if(
                                        forward_context_.kv_cache_vec.begin(), forward_context_.kv_cache_vec.end(),
                                        [](const infinilm::runtime::Tensor &t) { return static_cast<bool>(t); });
                                    if (cache == forward_context_.kv_cache_vec.end() || (*cache)->ndim() != 5
                                        || !effective_args.past_sequence_lengths || !effective_args.total_sequence_lengths) {
                                        throw std::invalid_argument("Static prefill requires an ordinary 5D KV cache and CPU lengths");
                                    }
                                    const auto capacity = (*cache)->size(1);
                                    const auto page_size = (*cache)->size(3);
                                    for (const auto &layer_cache : forward_context_.kv_cache_vec) {
                                        if (layer_cache && (layer_cache->ndim() != 5 || layer_cache->size(1) != capacity || layer_cache->size(3) != page_size)) {
                                            throw std::invalid_argument("Static prefill requires matching cache row layouts");
                                        }
                                    }
                                    forward_context_.attn_metadata.static_prefill_host = describe_static_prefill(
                                        batch, query_length, capacity, page_size,
                                        *effective_args.past_sequence_lengths,
                                        *effective_args.total_sequence_lengths,
                                        effective_args.input_offsets, local_args.cu_seqlens,
                                        effective_args.block_tables, local_args.slot_mapping);
                                }
                                auto model_output = model_->forward(model_args, forward_context_);
                                logits = model_output.logits;
                                hidden_states = model_output.hidden_states;
                            }

                            if (rank_info_.pp_size > 1 && rank_info_.pp_stage + 1 != rank_info_.pp_size) {
                                infinilm::runtime::Tensor output_ids;
                                if (rank_info_.pp_stage == 0 && rank_info_.tp_rank == 0) {
                                    // The last PP stage samples tokens. Return them
                                    // directly to stage-0/rank-0, which owns the
                                    // scheduler and user-facing request lifecycle.
                                    const size_t n_req = effective_args.input_offsets.value()->size(0) - 1;
                                    const auto *input_offsets = reinterpret_cast<const int32_t *>(effective_args.input_offsets.value()->data());
                                    const size_t n_out = local_args.sample_all_positions
                                                           ? static_cast<size_t>(input_offsets[n_req])
                                                           : n_req;
                                    if (runtime_->device() != rank_info_.device) {
                                        throw std::invalid_argument("Recv runtime and requested devices differ");
                                    }
                                    output_ids = infinilm::runtime::Tensor::empty(*runtime_, {n_out}, infinilm::runtime::DataType::kInt64);
                                    infinilm::engine::distributed::recv_(*runtime_, output_ids, (rank_info_.pp_size - 1) * rank_info_.tp_size, rank_info_.world_comm);
                                    output_ids = output_ids->to(*runtime_, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
                                    runtime_->syncStream();
                                }
                                output_ = Output{
                                    output_ids,
                                    logits,
                                    hidden_states,
                                    std::move(graph_execution),
                                    eager_reason};
                                job_done_ = true;
                                cv_.notify_all();
                                continue;
                            }

                            // Random sampling (rank 0 only)
                            if (rank_info_.tp_rank == 0) {
                                auto temperature{local_args.temperature};
                                auto top_p{local_args.top_p};
                                auto top_k{local_args.top_k};

                                const auto &logits_shape{logits->shape()};
                                const auto &vocab_size{logits_shape[2]};
                                const auto &total_len{logits_shape[1]};
                                const auto &batch_size{logits_shape[0]};

                                auto n_req = effective_args.input_offsets.value()->size(0) - 1;
                                int32_t *input_offsets = (int32_t *)effective_args.input_offsets.value()->data();

                                const bool sample_all_positions = local_args.sample_all_positions;
                                const size_t logits_positions = batch_size * total_len;
                                const bool logits_are_last_token_only = !sample_all_positions && logits_positions == n_req;
                                const size_t n_out = sample_all_positions ? static_cast<size_t>(input_offsets[n_req]) : n_req;
                                auto output_ids{infinilm::runtime::Tensor::empty(*runtime_, {n_out}, infinilm::runtime::DataType::kInt64)};

                                const bool parameter_greedy = local_args.sampling_rows.has_value()
                                                                ? std::all_of(local_args.sampling_rows->begin(),
                                                                              local_args.sampling_rows->end(),
                                                                              [](const auto &row) {
                                                                                  return std::get<1>(row) == 0.0f
                                                                                      || std::get<2>(row) == 1
                                                                                      || std::get<0>(row) == 0.0f;
                                                                              })
                                                                : top_p == 0.0f || top_k == 1 || temperature == 0.0f;
                                const bool batch_greedy = parameter_greedy
                                                       && n_out > 0
                                                       && logits_positions == n_out
                                                       && logits->is_contiguous()
                                                       && (sample_all_positions || logits_are_last_token_only);
                                if (batch_greedy) {
                                    float random_val = 0.0f;
                                    if (!local_args.random_values.has_value()) {
                                        for (size_t i{0}; i < n_out; ++i) {
                                            random_val = std::uniform_real_distribution<float>(0, 1)(rng_);
                                        }
                                    }
                                    sample_request_token(
                                        *runtime_,
                                        output_ids,
                                        logits->view({logits_positions, vocab_size}),
                                        random_val,
                                        1.0f,
                                        1,
                                        1.0f);
                                } else {
                                    for (size_t i{0}; i < n_out; ++i) {
                                        size_t score_idx = i;
                                        if (!sample_all_positions && !logits_are_last_token_only) {
                                            score_idx = static_cast<size_t>(input_offsets[i + 1] - 1);
                                        }
                                        auto score{logits->view({logits_positions, vocab_size})->narrow({{0, score_idx, 1}})->view({vocab_size})};
                                        auto out{output_ids->narrow({{0, i, 1}})->view({})};
                                        float random_val = local_args.random_values.has_value()
                                                             ? local_args.random_values.value()[i]
                                                             : std::uniform_real_distribution<float>(0, 1)(rng_);
                                        if (local_args.sampling_rows.has_value()) {
                                            std::tie(temperature, top_p, top_k) = local_args.sampling_rows.value()[i];
                                        }
                                        sample_request_token(
                                            *runtime_,
                                            out, score, random_val, top_p, top_k, temperature);
                                    }
                                }

                                if (rank_info_.pp_size > 1) {
                                    infinilm::engine::distributed::send(
                                        *runtime_,
                                        output_ids,
                                        0,
                                        rank_info_.world_comm);
                                }

                                output_ids = output_ids->to(*runtime_, infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});

                                runtime_->syncStream();
                                if (nsa_pool_epoch_) {
                                    model_->finish_paged_owner(true);
                                    forward_context_.nsa_owner.reset();
                                    nsa_drain_required_ = false;
                                }

                                auto out{Output{output_ids, logits, hidden_states,
                                                std::move(graph_execution), eager_reason}};

                                output_ = std::move(out);
                            }

                            if (nsa_pool_epoch_ && rank_info_.tp_rank != 0) {
                                model_->finish_paged_owner(false); // host success remains Queued, never Completed
                                forward_context_.nsa_owner.reset();
                            }
                            job_done_ = true;
                        } catch (...) {
                            const auto run_error = std::current_exception(); // active catch, never destructor
                            if (nsa_pool_epoch_) {
                                nsa::retire_caught_run_failure(run_error, [&](std::exception_ptr error) {
                                        if (barrier_){ barrier_->abort(error);
} }, [&] {
                                        model_->invalidate_paged_owner();
                                        forward_context_.nsa_owner.reset(); }, [&] { return drain_nsa_owner_noexcept(); });
                            }
                            throw; // ordinary model path preserves its existing outer catch
                        }
                    }
                    cv_.notify_all();

                } catch (...) {
                    const auto error = std::current_exception();
                    if (nsa_pool_epoch_) {
                        if (barrier_) {
                            barrier_->abort(error); // unblock peers before owner drain
                        }
                        model_->invalidate_paged_owner(); // validity only; releases no Tensor
                        forward_context_.nsa_owner.reset();
                        (void)drain_nsa_owner_noexcept();
                    }
                    fail("forward", error); // original error stays primary
                    break;
                }
            } else if (local_cmd == Command::RESET_CACHE) {
                try {
                    if (model_config_->get<std::string>("model_type") == "kimi_k3") {
                        runtime_->syncStream();
                    }
                    if (nsa_pool_epoch_) {
                        model_->invalidate_paged_owner();
                        if (!drain_nsa_owner_noexcept()) {
                            throw std::runtime_error("NSA reset refused after owner drain failure");
                        }
                        nsa_drain_required_ = true; // covers new allocation/initialization work
                    }
                    compiler_.reset();
                    static_decode_metadata_.reset();
                    finite_paged_decode_profile_.reset();
                    model_->reset_cache(local_cache_config != nullptr ? local_cache_config.get() : nullptr,
                                        forward_context_, attention_backend_, rank_info_);
                    refresh_finite_paged_decode_profile();
                    if (enable_graph_compiling_ && rank_info_.pp_size == 1) {
                        PagedGraphConfig{attention_backend_, rank_info_.tp_size, rank_info_.tp_rank, rank_info_.pp_size}.validate();
                    }
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        if (nsa_pool_epoch_) {
                            nsa_pool_epoch_ = local_nsa_pool_epoch;
                        }
                        if (model_config_->get<std::string>("model_type") == "kimi_k3") {
                            runtime_->syncStream();
                        }
                        job_done_ = true;
                    }
                    cv_.notify_all();

                } catch (...) {
                    const auto error = std::current_exception();
                    if (nsa_pool_epoch_) {
                        if (barrier_) {
                            barrier_->abort(error);
                        }
                        model_->invalidate_paged_owner();
                        (void)drain_nsa_owner_noexcept();
                    }
                    fail("reset_cache", error);
                    break;
                }
            } else if (local_cmd == Command::COMPILE) {
                try {
                    if (nsa_pool_epoch_) {
                        throw std::invalid_argument("NSA COMPILE rejected before compiler/cache touch");
                    }
                    if (enable_graph_compiling_ && rank_info_.pp_size == 1) {
                        compiler_.reset();
                        if (dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())) {
                            compiler_ = std::make_unique<StaticBatchingCompiler>(model_, barrier_, *runtime_, forward_context_);
                        } else if (dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
                            compiler_ = std::make_unique<PagedCompiler>(
                                model_, barrier_, *runtime_, forward_context_,
                                PagedGraphConfig{attention_backend_, rank_info_.tp_size, rank_info_.tp_rank, rank_info_.pp_size});
                        }
                        if (compiler_) {
                            compiler_->compile();
                        }
                    }
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        job_done_ = true;
                    }
                    cv_.notify_all();

                } catch (...) {
                    fail("compile", std::current_exception());
                    break;
                }

            } else {
                // Shouldn't reach here (no-op)
            }
        } // while
        // Some clean up should be done before exiting the thread
        if (runtime_ && model_config_->get<std::string>("model_type") == "kimi_k3") {
            runtime_->syncStream();
        }
        if (nsa_pool_epoch_ && model_) {
            model_->invalidate_paged_owner();
        }
        if (drain_nsa_owner_noexcept()) {
            compiler_.reset();
        }
    } catch (...) {
        const auto error = std::current_exception();
        if (nsa_pool_epoch_) {
            if (barrier_) {
                barrier_->abort(error);
            }
            if (model_) {
                model_->invalidate_paged_owner();
            }
            (void)drain_nsa_owner_noexcept();
        }
        fail("thread_loop", error);
    }
}

} // namespace infinilm::engine
