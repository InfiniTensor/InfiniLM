#include "rank_worker.hpp"
#include "../models/model_factory.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/send_recv.hpp"
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/equal.hpp>
#include <infinicore/ops/mul.hpp>
#include <spdlog/spdlog.h>
#include <stdexcept>

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
// close -- request shutdown and join thread
//------------------------------------------------------
std::vector<std::vector<infinicore::Tensor>> RankWorker::get_hybrid_states() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return init_done_ || should_exit_; });
    if (should_exit_ || has_job_) {
        throw std::runtime_error("State access requires an idle worker.");
    }
    return {forward_context_.conv_state_vec, forward_context_.ssm_state_vec};
}

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

//------------------------------------------------------
// thread_loop
//------------------------------------------------------
void RankWorker::thread_loop() {
    try {
        {
            std::lock_guard<std::mutex> lk(mutex_);

            // Initialize device & model outside of holding the main mutex to avoid blocking callers.
            infinicore::context::setDevice(rank_info_.device);

            // Initialize global enviromnet.
            infinilm::global_state::initialize_model_parallel(rank_info_);
            infinilm::global_state::initialize_forward_context(forward_context_);
            infinilm::global_state::initialize_infinilm_config(infinilm_config_);

            // Create model using factory (may be expensive)
            model_ = InfinilmModelFactory::createModel(
                model_config_,
                rank_info_.device,
                pending_cache_config_ != nullptr ? pending_cache_config_.get() : nullptr);
            if (enable_graph_compiling_ && rank_info_.pp_size == 1) {
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
                        std::lock_guard<std::mutex> lk(mutex_);

                        infinicore::Tensor logits;
                        infinicore::Tensor hidden_states;
                        infinicore::Tensor sampled_ids;
                        infinicore::Tensor model_input_ids;
                        if (local_args.token_state_indices.has_value()
                            && !model_->supports_token_state_checkpoints()) {
                            throw std::runtime_error("This model does not support per-token state checkpoints.");
                        }
                        const bool graph_candidate = !local_args.token_state_indices
                                                  && local_args.input_ids && local_args.input_offsets
                                                  && (local_args.input_ids.value()->numel() == local_args.input_offsets.value()->numel() - 1
                                                      || (local_args.target_hidden_states && local_args.input_ids.value()->numel() <= 2));
                        if (graph_candidate && compiler_ != nullptr && rank_info_.pp_size == 1) {
                            auto graph_input = local_args.to_model_input(infinicore::Device::cpu(), true);
                            auto [graph, output] = compiler_->get_compiled(graph_input);
                            if (graph != nullptr && output != nullptr) {
                                graph->run();
                                logits = output->logits;
                                hidden_states = output->hidden_states;
                                model_input_ids = graph_input.input_ids.value();
                            }
                        }
                        // Fall back to eager mode
                        if (!logits) {
                            auto model_args = local_args.to_model_input(rank_info_.device);
                            model_input_ids = model_args.input_ids.value();
                            auto model_output = model_->forward(model_args);
                            logits = model_output.logits;
                            hidden_states = model_output.hidden_states;
                            sampled_ids = model_output.output_ids;
                        }

                        if (rank_info_.pp_size > 1 && rank_info_.pp_stage + 1 != rank_info_.pp_size) {
                            infinicore::Tensor output_ids;
                            if (rank_info_.pp_stage == 0 && rank_info_.tp_rank == 0) {
                                // The last PP stage samples tokens. Return them
                                // directly to stage-0/rank-0, which owns the
                                // scheduler and user-facing request lifecycle.
                                const size_t n_req = local_args.input_offsets.value()->size(0) - 1;
                                const auto *input_offsets = reinterpret_cast<const int32_t *>(local_args.input_offsets.value()->data());
                                const size_t n_out = local_args.sample_all_positions
                                                       ? static_cast<size_t>(input_offsets[n_req])
                                                       : n_req;
                                output_ids = infinicore::op::distributed::recv(
                                    {n_out},
                                    infinicore::DataType::I64,
                                    rank_info_.device,
                                    (rank_info_.pp_size - 1) * rank_info_.tp_size,
                                    rank_info_.world_comm);
                                output_ids = output_ids->to(infinicore::Device::cpu());
                                infinicore::context::syncStream();
                            }
                            output_ = Output{
                                output_ids,
                                logits,
                                hidden_states};
                            job_done_ = true;
                            cv_.notify_all();
                            continue;
                        }

                        // Random sampling (rank 0 only)
                        if (rank_info_.tp_rank == 0) {
                            auto output_ids = sampled_ids;
                            if (!output_ids) {
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
                                const size_t logits_positions = batch_size * total_len;
                                const bool logits_are_last_token_only = !sample_all_positions && logits_positions == n_req;
                                const size_t n_out = sample_all_positions ? static_cast<size_t>(input_offsets[n_req]) : n_req;
                                output_ids = infinicore::Tensor::empty({n_out}, infinicore::DataType::I64, rank_info_.device);

                                for (size_t i{0}; i < n_out; ++i) {
                                    size_t score_idx = i;
                                    if (!sample_all_positions && !logits_are_last_token_only) {
                                        score_idx = static_cast<size_t>(input_offsets[i + 1] - 1);
                                    }
                                    auto score{logits->view({logits_positions, vocab_size})->narrow({{0, score_idx, 1}})->view({vocab_size})};
                                    auto out{output_ids->narrow({{0, i, 1}})->view({})};
                                    float random_val = std::uniform_real_distribution<float>(0, 1)(rng_);
                                    infinicore::op::random_sample_(
                                        out, score, random_val, top_p, top_k, temperature);
                                }
                            }

                            if (rank_info_.pp_size > 1) {
                                infinicore::op::distributed::send(
                                    output_ids,
                                    0,
                                    rank_info_.world_comm);
                            }

                            int accepted_draft_tokens = -1;
                            if (local_args.verify_draft) {
                                const size_t count = model_input_ids->size(1) - 1;
                                auto candidates = model_input_ids->narrow({{1, 1, count}})->view({count})->to(rank_info_.device);
                                auto accepted = infinicore::op::equal(output_ids->narrow({{0, 0, count}}), candidates);
                                auto packed = infinicore::Tensor::empty({count + 2}, infinicore::DataType::I64, rank_info_.device);
                                packed->narrow({{0, 0, count + 1}})->copy_from(output_ids);
                                auto length = packed->narrow({{0, count + 1, 1}});
                                if (count == 1) {
                                    infinicore::op::cast_(length, accepted);
                                } else {
                                    // Sum consecutive prefix matches, stopping at the
                                    // first rejection. F32 represents these 0..4 counts exactly.
                                    auto matches = infinicore::Tensor::empty({count}, infinicore::DataType::F32, rank_info_.device);
                                    infinicore::op::cast_(matches, accepted);
                                    auto prefix = matches->narrow({{0, 0, 1}});
                                    auto total = prefix;
                                    for (size_t i = 1; i < count; ++i) {
                                        prefix = infinicore::op::mul(prefix, matches->narrow({{0, i, 1}}));
                                        total = infinicore::op::add(total, prefix);
                                    }
                                    infinicore::op::cast_(length, total);
                                }
                                // One bounded host transfer contains the tokens and
                                // acceptance length. Scheduler ownership stays on CPU.
                                infinicore::context::syncStream();
                                auto host = packed->to(infinicore::Device::cpu());
                                accepted_draft_tokens = static_cast<int>(reinterpret_cast<int64_t *>(host->data())[count + 1]);
                                output_ids = host->narrow({{0, 0, static_cast<size_t>(1 + accepted_draft_tokens)}});
                            } else if (!local_args.return_device_tokens) {
                                infinicore::context::syncStream();
                                output_ids = output_ids->to(infinicore::Device::cpu());
                            }

                            auto out{Output{output_ids, logits, hidden_states, accepted_draft_tokens}};

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
                        spdlog::info("Graph capture begin: tp_rank={}", rank_info_.tp_rank);
                        compiler_->compile();
                        spdlog::info("Graph capture end: tp_rank={}", rank_info_.tp_rank);
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
        // Some clean up should be done before exiting the thread
        compiler_.reset();
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
