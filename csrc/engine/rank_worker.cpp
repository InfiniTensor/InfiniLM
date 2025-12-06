#include "rank_worker.hpp"

#include "../models/model_factory.hpp"
#include "../models/llama/llama_for_causal_lm.hpp"

#include <iostream>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::engine {

RankWorker::RankWorker(const std::any &model_config,
                       const distributed::RankInfo &rank_info)
    : model_config_(model_config),
      rank_info_(rank_info),
      job_cmd_(Command::INIT),
      has_job_(false),
      job_done_(false),
      should_exit_(false),
      init_done_(false) {
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
// run -- asynchronous
//------------------------------------------------------
void RankWorker::run(const std::vector<std::any> &args) {
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
// wait -- asynchronous
//------------------------------------------------------
void RankWorker::wait() {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return job_done_ || should_exit_; });

    if (should_exit_) {
        throw std::runtime_error("RankWorker stopped during run");
    }
}

//------------------------------------------------------
// reset_cache -- synchronous (blocks until worker finishes reset)
//------------------------------------------------------
void RankWorker::reset_cache(bool full_reset) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (should_exit_) {
            throw std::runtime_error("RankWorker is closing; cannot reset_cache");
        }

        pending_full_reset_ = full_reset;
        job_cmd_ = Command::RESET_CACHE;
        has_job_ = true;
        job_done_ = false;
    }
    cv_.notify_all();

    // Wait for job completion
    std::unique_lock<std::mutex> lk(mutex_);
    cv_.wait(lk, [&] { return job_done_ || should_exit_; });

    if (should_exit_) {
        throw std::runtime_error("RankWorker stopped while resetting cache");
    }
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
infinicore::Tensor RankWorker::get_output() {
    std::lock_guard<std::mutex> lock(mutex_);
    return output_;
}

//------------------------------------------------------
// thread_loop
//------------------------------------------------------
void RankWorker::thread_loop() {
    try {
        // Initialize device & model outside of holding the main mutex to avoid blocking callers.
        infinicore::context::setDevice(rank_info_.device);

        // Create model using factory (may be expensive)
        model_ = InfinilmModelFactory::createModel(model_config_, rank_info_);

        // Signal that initialization is done
        {
            std::lock_guard<std::mutex> lk(mutex_);
            init_done_ = true;
        }
        cv_.notify_all();

        // Main loop: wait for jobs or exit
        while (true) {
            Command local_cmd = Command::INIT;
            std::string local_param_name;
            infinicore::Tensor local_param;
            std::vector<std::any> local_args;
            bool local_full_reset = true;

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
                } else if (local_cmd == Command::RUN) {
                    local_args = pending_args_;
                } else if (local_cmd == Command::RESET_CACHE) {
                    local_full_reset = pending_full_reset_;
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
                    // convert exceptions to a safe behavior: set should_exit_ and notify caller
                    std::lock_guard<std::mutex> lk(mutex_);
                    should_exit_ = true;
                    job_done_ = true;
                    cv_.notify_all();
                    // rethrow so the thread can be joined and caller sees an error if desired (optional)
                    spdlog::error("[{}] exception during load_parameter_: {}\n", info(), e.what());
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
                    auto out = model_->forward(local_args);

                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        output_ = std::move(out);
                        job_done_ = true;
                    }
                    cv_.notify_all();

                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lk(mutex_);
                    should_exit_ = true;
                    job_done_ = true;
                    cv_.notify_all();
                    spdlog::error("[{}] exception during forward: {}\n", info(), e.what());
                    break;
                }
            } else if (local_cmd == Command::RESET_CACHE) {
                try {
                    // Cast model to LlamaForCausalLM to access reset_cache
                    // Since we know it's LlamaForCausalLM for llama models, we can cast
                    auto* llama_model = dynamic_cast<models::llama::LlamaForCausalLM*>(model_.get());
                    if (llama_model) {
                        llama_model->model().reset_cache(local_full_reset);
                    }

                    {
                        std::lock_guard<std::mutex> lk(mutex_);
                        job_done_ = true;
                    }
                    cv_.notify_all();

                } catch (const std::exception &e) {
                    std::lock_guard<std::mutex> lk(mutex_);
                    should_exit_ = true;
                    job_done_ = true;
                    cv_.notify_all();
                    spdlog::error("[{}] exception during reset_cache: {}\n", info(), e.what());
                    break;
                }
            } else {
                // Shouldn't reach here (no-op)
            }
        } // while
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
