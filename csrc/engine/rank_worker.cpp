#include "rank_worker.hpp"
#include "topology/device_topology.hpp"
#include "../utils.hpp"
#include "../models/model_factory.hpp"
#include "workspace/workspace_context.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <tuple>

namespace infinilm::engine {

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
}

void RankWorker::wait_for_init() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });
    if (should_exit_) {
        throw std::runtime_error("RankWorker failed to initialize");
    }
}

RankWorker::~RankWorker() {
    close();
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
// load_params -- synchronous batch load
//------------------------------------------------------
void RankWorker::load_params(const std::unordered_map<std::string, infinicore::Tensor> &params, bool strict) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (should_exit_) {
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

std::vector<std::string> RankWorker::state_dict_keys() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });

    if (!model_) {
        throw std::runtime_error("state_dict_keys called before model initialization");
    }

    return model_->state_dict_keys();
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
// get kv cache
//------------------------------------------------------
std::vector<infinicore::Tensor> RankWorker::get_kv_cache() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });

    if (should_exit_) {
        throw std::runtime_error("RankWorker stopped; cannot get_cache_vec");
    }

    ASSERT(forward_context_.kv_cache_vec.size() > 0 && "RankWorker::get_kv_cache(): kv_cache_vec is empty");

    return forward_context_.kv_cache_vec;
}

//------------------------------------------------------
// request_close -- request shutdown without waiting for the thread
//------------------------------------------------------
void RankWorker::request_close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        should_exit_ = true;
        has_job_ = false; // don't keep old jobs pending
        job_cmd_ = Command::STOP;
    }
    cv_.notify_all();
}

//------------------------------------------------------
// join -- wait for worker thread shutdown
//------------------------------------------------------
void RankWorker::join() {
    if (thread_.joinable()) {
        thread_.join();
    }
}

//------------------------------------------------------
// close -- request shutdown and join thread
//------------------------------------------------------
void RankWorker::close() {
    request_close();
    join();
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
            infinicore::context::setDevice(rank_info_.device);
            auto affinity_binding = topology::bind_current_thread_to_device_numa(rank_info_.device);
            if (affinity_binding.applied) {
                spdlog::info(
                    "{} bound worker thread to NUMA node {} CPUs {} via {}{}",
                    rank_info_.to_string(),
                    affinity_binding.numa_node,
                    affinity_binding.cpu_list,
                    affinity_binding.provider,
                    affinity_binding.pci_bus_id.empty() ? "" : " pci=" + affinity_binding.pci_bus_id);
            } else if (affinity_binding.attempted) {
                spdlog::debug(
                    "{} skipped worker CPU affinity binding via {}: {}",
                    rank_info_.to_string(),
                    affinity_binding.provider,
                    affinity_binding.reason);
            }
            workspace_manager_ = std::make_unique<InferenceWorkspaceManager>(rank_info_.device);
            const bool enable_async_collectives = distributed::async_collectives_enabled_for_rank(
                rank_info_.device,
                rank_info_.tp_size,
                rank_info_.comm);
            async_collective_context_ = std::make_unique<distributed::AsyncCollectiveContext>(
                rank_info_.device,
                enable_async_collectives);
            if (async_collective_context_->enabled()) {
                spdlog::info("{} enabled async collective context", rank_info_.to_string());
            }

            // Initialize global enviromnet.
            infinilm::global_state::initialize_model_parallel(rank_info_);
            infinilm::global_state::initialize_forward_context(forward_context_);
            infinilm::global_state::initialize_infinilm_config(infinilm_config_);

            // Create model using factory (may be expensive)
            model_ = InfinilmModelFactory::createModel(
                model_config_,
                rank_info_.device,
                pending_cache_config_ != nullptr ? pending_cache_config_.get() : nullptr);
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
            std::unordered_map<std::string, infinicore::Tensor> local_params;
            bool local_params_strict = true;
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

            } else if (local_cmd == Command::LOAD_BATCH) {
                try {
                    model_->load_parameters_no_sync(local_params, local_params_strict);
                    infinicore::context::syncStream();
                } catch (const std::exception &e) {
                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        should_exit_ = true;
                        job_done_ = true;
                    }
                    cv_.notify_all();
                    spdlog::error("[{}] exception during load_parameters_: {}\n", info(), e.what());
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
                    model_->process_weights_after_loading();
                    infinicore::context::syncStream();
                    infinicore::context::trimMemory();
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
                    {
                        WorkspaceForwardGuard forward_guard(workspace_manager_.get());
                        WorkspaceContextGuard workspace_guard(workspace_manager_.get());
                        distributed::AsyncCollectiveContextGuard async_collective_guard(async_collective_context_.get());
                        std::lock_guard<std::mutex> lk(mutex_);

                        infinicore::Tensor logits;
                        infinicore::Tensor hidden_states;
                        if (local_args.wait_event) {
                            infinicore::context::setDevice(rank_info_.device);
                            infinicore::context::streamWaitEvent(
                                infinicore::context::getStream(), local_args.wait_event->get());
                        }
                        auto model_args = local_args.to_model_input(rank_info_.device);
                        std::shared_ptr<infinicore::graph::Graph> graph;
                        std::shared_ptr<InfinilmModel::Output> graph_output;
                        if (compiler_ != nullptr && !local_args.sample_all_positions) {
                            std::tie(graph, graph_output) = compiler_->get_compiled(model_args);
                        }

                        const bool use_compiled_decode_graph = graph != nullptr && graph_output != nullptr;
                        if (rank_info_.comm != nullptr &&
                            rank_info_.allreduce_backend == INFINICCL_ALLREDUCE_BACKEND_CUSTOM) {
                            RUN_INFINI(infinicclCommSetAllReduceBackend(
                                rank_info_.comm,
                                use_compiled_decode_graph ? INFINICCL_ALLREDUCE_BACKEND_CUSTOM
                                                          : INFINICCL_ALLREDUCE_BACKEND_NCCL));
                        }

                        if (use_compiled_decode_graph) {
                            graph->run();
                            logits = graph_output->logits;
                        }
                        // Fall back to eager mode. This covers prefill and unsupported decode shapes;
                        // when custom was requested, it is forced to NCCL above for this eager path.
                        if (!logits) {
                            auto model_output = model_->forward(model_args);
                            logits = model_output.logits;
                            hidden_states = model_output.hidden_states;
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

                            auto n_req = local_args.input_offsets.value()->size(0) - 1;
                            int32_t *input_offsets = (int32_t *)local_args.input_offsets.value()->data();

                            const bool sample_all_positions = local_args.sample_all_positions;
                            const size_t n_out = sample_all_positions
                                                     ? static_cast<size_t>(input_offsets[n_req])
                                                     : n_req;
                            const auto output_dtype = sample_all_positions
                                                          ? infinicore::DataType::I64
                                                          : infinicore::DataType::I32;
                            auto output_ids{infinicore::Tensor::empty({n_out}, output_dtype, rank_info_.device)};
                            for (size_t i{0}; i < n_out; ++i) {
                                size_t score_idx = i;
                                if (!sample_all_positions) {
                                    score_idx = static_cast<size_t>(input_offsets[i + 1] - 1);
                                }
                                auto score{logits->view({batch_size * total_len, vocab_size})->narrow({{0, score_idx, 1}})->view({vocab_size})};
                                auto out{output_ids->narrow({{0, i, 1}})->view({})};
                                float random_val = std::uniform_real_distribution<float>(0, 1)(rng_);
                                infinicore::op::random_sample_(
                                    out, score, random_val, top_p, top_k, temperature);
                            }

                            auto ready_event = std::make_shared<infinicore::DeviceEvent>(rank_info_.device);
                            ready_event->record(infinicore::context::getStream());
                            auto out{Output{output_ids, logits, hidden_states, ready_event}};
                            output_ = std::move(out);
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
                    if (compiler_ != nullptr) {
                        WorkspaceContextGuard workspace_guard(workspace_manager_.get());
                        distributed::AsyncCollectiveContextGuard async_collective_guard(async_collective_context_.get());
                        compiler_->compile();
                    }
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
        // Release graph/model-owned GPU resources on the worker's CUDA context
        // instead of leaving them to Python interpreter shutdown.
        infinicore::context::setDevice(rank_info_.device);
        infinicore::context::syncStream();
        {
            std::lock_guard<std::mutex> lk(mutex_);
            output_ = Output{};
            pending_args_ = Input{};
        }
        compiler_.reset();
        model_.reset();
        workspace_manager_.reset();
        async_collective_context_.reset();
        infinicore::context::syncStream();
    } catch (const std::exception &e) {
        // Top-level exception: ensure any waiters are woken and the thread exits cleanly.
        {
            std::lock_guard<std::mutex> lk(mutex_);
            init_done_ = true;
            should_exit_ = true;
            job_done_ = true;
        }
        cv_.notify_all();
        spdlog::error("[{}] fatal exception in thread_loop: {} \n", info(), e.what());
    }
}

} // namespace infinilm::engine
