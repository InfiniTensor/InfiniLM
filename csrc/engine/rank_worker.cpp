#include "rank_worker.hpp"
#include "../models/model_factory.hpp"
#include "../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::engine {
namespace {

infinicore::Tensor logits_to_cpu_f32(const infinicore::Tensor &logits) {
    auto cpu = logits->contiguous()->to(infinicore::Device::cpu());
    auto out = infinicore::Tensor::empty(cpu->shape(), infinicore::DataType::F32, infinicore::Device::cpu());
    auto *dst = reinterpret_cast<float *>(out->data());
    const size_t count = cpu->numel();
    switch (cpu->dtype()) {
    case infinicore::DataType::F32:
        std::memcpy(dst, cpu->data(), count * sizeof(float));
        break;
    case infinicore::DataType::F16: {
        const auto *src = reinterpret_cast<const uint16_t *>(cpu->data());
        for (size_t i = 0; i < count; ++i) {
            dst[i] = f16_to_f32(src[i]);
        }
        break;
    }
    case infinicore::DataType::BF16: {
        const auto *src = reinterpret_cast<const uint16_t *>(cpu->data());
        for (size_t i = 0; i < count; ++i) {
            dst[i] = bf16_to_f32(src[i]);
        }
        break;
    }
    default:
        throw std::runtime_error("RankWorker: unsupported logits dtype for debug output");
    }
    return out;
}

bool cpu_greedy_sample_enabled() {
    const char *value = std::getenv("INFINILM_ENABLE_CPU_GREEDY_SAMPLE");
    return value != nullptr && std::string(value) == "1";
}

infinicore::Tensor greedy_sample_to_cpu_i64(const infinicore::Tensor &logits,
                                            const infinicore::Tensor &input_offsets) {
    auto logits_cpu = logits_to_cpu_f32(logits);
    const auto &shape = logits_cpu->shape();
    const size_t vocab_size = shape[2];
    const size_t total_len = shape[1];
    const size_t batch_size = shape[0];
    auto offsets_cpu = input_offsets->contiguous()->to(infinicore::Device::cpu());
    const size_t n_req = offsets_cpu->size(0) - 1;
    const auto *offsets = reinterpret_cast<const int32_t *>(offsets_cpu->data());
    auto output_ids = infinicore::Tensor::empty({n_req}, infinicore::DataType::I64, infinicore::Device::cpu());
    auto *out = reinterpret_cast<int64_t *>(output_ids->data());
    const auto *values = reinterpret_cast<const float *>(logits_cpu->data());
    for (size_t i = 0; i < n_req; ++i) {
        const size_t row = static_cast<size_t>(offsets[i + 1] - 1);
        if (row >= batch_size * total_len) {
            throw std::runtime_error("RankWorker: input_offsets out of logits range");
        }
        const float *score = values + row * vocab_size;
        size_t best = 0;
        float best_value = std::isfinite(score[0]) ? score[0] : -std::numeric_limits<float>::infinity();
        for (size_t token = 1; token < vocab_size; ++token) {
            const float value = score[token];
            if (std::isfinite(value) && value > best_value) {
                best = token;
                best_value = value;
            }
        }
        if (!std::isfinite(best_value)) {
            best = 0;
        }
        out[i] = static_cast<int64_t>(best);
    }
    return output_ids;
}

bool tensor_parallel_comm_warmup_enabled() {
    const char *value = std::getenv("INFINILM_TP_COMM_WARMUP");
    return value == nullptr || std::string(value) != "0";
}

void warmup_tensor_parallel_allreduce(const distributed::RankInfo &rank_info) {
    if (!tensor_parallel_comm_warmup_enabled()
        || rank_info.tp_size <= 1
        || rank_info.comm == nullptr
        || rank_info.device.getType() == infinicore::Device::Type::CPU) {
        return;
    }

    constexpr size_t warmup_elems = 1 << 20;
    auto f32 = infinicore::Tensor::empty({warmup_elems}, infinicore::DataType::F32, rank_info.device);
    infinicore::op::distributed::allreduce_(f32, f32, INFINICCL_SUM, rank_info.comm);

    auto bf16 = infinicore::Tensor::empty({warmup_elems}, infinicore::DataType::BF16, rank_info.device);
    infinicore::op::distributed::allreduce_(bf16, bf16, INFINICCL_SUM, rank_info.comm);
    infinicore::context::syncStream();
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
    // Start the thread; InferEngine waits on all workers via wait_init() so ranks init in parallel.
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
                    warmup_tensor_parallel_allreduce(rank_info_);
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
                        // All-position speculative/MTP runs need eager mode because
                        // hidden states are not part of compiled graph outputs.
                        if (!local_args.sample_all_positions && compiler_ != nullptr) {
                            auto [graph, output] = compiler_->get_compiled(local_args.to_model_input(infinicore::Device::cpu()));
                            if (graph != nullptr && output != nullptr) {
                                graph->run();
                                logits = output->logits;
                            }
                        }
                        // Fall back to eager mode
                        if (!logits) {
                            auto model_args = local_args.to_model_input(rank_info_.device);
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
                            const char *dump_logits_env = std::getenv("INFINILM_DUMP_DSV4_LOGITS");
                            const bool dump_any_logits = dump_logits_env != nullptr && std::string(dump_logits_env) == "1";
                            const std::string dump_stage = total_len == 1 ? "decode" : "prefill";

                            if (dump_any_logits) {
                                auto logits_cpu = logits_to_cpu_f32(logits);
                                const auto &dump_shape = logits_cpu->shape();
                                const std::string dump_base = "/tmp/infinilm_dsv4_layer0_" + dump_stage + "_logits";
                                std::ofstream meta(dump_base + ".json");
                                meta << "{\"shape\":[";
                                for (size_t i = 0; i < dump_shape.size(); ++i) {
                                    if (i != 0) {
                                        meta << ",";
                                    }
                                    meta << dump_shape[i];
                                }
                                meta << "],\"dtype\":\"float32\",\"stage\":\"" << dump_stage << "\"}";

                                std::ofstream data(dump_base + ".txt");
                                data << std::setprecision(9);
                                const auto *vals = reinterpret_cast<const float *>(logits_cpu->data());
                                for (size_t i = 0; i < logits_cpu->numel(); ++i) {
                                    data << vals[i] << '\n';
                                }
                                spdlog::info("RankWorker dumped {} logits to {}", dump_stage, dump_base);
                            }

                            auto n_req = local_args.input_offsets.value()->size(0) - 1;
                            int32_t *input_offsets = (int32_t *)local_args.input_offsets.value()->data();

                            const bool sample_all_positions = local_args.sample_all_positions;
                            infinicore::Tensor output_ids;
                            const bool greedy = cpu_greedy_sample_enabled()
                                             && !sample_all_positions
                                             && (top_k == 1 || temperature == 0.0f || top_p == 0.0f);
                            if (greedy) {
                                output_ids = greedy_sample_to_cpu_i64(logits, local_args.input_offsets.value());
                            } else {
                                const size_t n_out = sample_all_positions
                                                       ? static_cast<size_t>(input_offsets[n_req])
                                                       : n_req;
                                output_ids = infinicore::Tensor::empty({n_out}, infinicore::DataType::I64, rank_info_.device);
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
                                if (!sample_all_positions) {
                                    output_ids = output_ids->to(infinicore::Device::cpu());
                                }
                            }

                            infinicore::context::syncStream();

                            auto out{Output{output_ids, logits, hidden_states}};

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
