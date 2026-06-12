#include "infer_engine.hpp"
#include "../config/config_factory.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <stdexcept>
#include <string>

namespace infinilm::engine {

//------------------------------------------------------
// Constructor
//------------------------------------------------------
InferEngine::InferEngine(
    const std::string &config_str,
    const distributed::DistConfig &distributed_config,
    infinicore::Device::Type device_type,
    const cache::CacheConfig *cache_config,
    bool enable_graph_compiling,
    backends::AttentionBackend attention_backend,
    std::optional<infinicore::DataType> kv_cache_dtype,
    const std::string &weight_load_mode) // Changed parameter
    : communication_group_(distributed_config, device_type),
      attention_backend_(attention_backend),
      weight_load_mode_(weight_load_mode),
      weight_load_group_size_(2),
      weight_load_clone_(weight_load_mode == "grouped-clone") {
    if (weight_load_mode_ != "sync" && weight_load_mode_ != "async" && weight_load_mode_ != "grouped" && weight_load_mode_ != "grouped-clone") {
        throw std::invalid_argument("weight_load_mode must be one of: sync, async, grouped, grouped-clone");
    }
    if (cache_config != nullptr) {
        cache_config_ = cache_config->unique_copy();
    }

    // Load model config if model_path is provided, model_path must be valid, and config.json exists
    this->model_config_ = infinilm::config::ConfigFactory::createConfig(config_str);
    auto infinilm_config = std::make_shared<infinilm::global_state::InfinilmConfig>(attention_backend, this->model_config_);

    // Only support offline int8 kv cache quantization in this version
    if (kv_cache_dtype.has_value()) {
        this->model_config_->set_kv_quant_scheme(kv_cache_dtype.value());
    }
    // Create one RankWorker per rank
    int world_size = communication_group_.get_world_size();
    barrier_ = std::make_unique<RankBarrier>((size_t)world_size);
    workers_.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
        workers_.emplace_back(std::make_unique<RankWorker>(
            infinilm_config,
            communication_group_.get_rank_info(r),
            cache_config_ != nullptr ? cache_config_.get() : nullptr,
            barrier_.get(),
            enable_graph_compiling,
            attention_backend_));
    }
    // Compile the model on all workers
    this->compile();
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::load_param(const std::string &name, const infinicore::Tensor &param) {
    // Load the parameter on all workers
    for (auto &worker : workers_) {
        worker->load_param(name, param);
    }
}

void InferEngine::load_params(const std::unordered_map<std::string, infinicore::Tensor> &params) {
    if (workers_.size() <= 1 || weight_load_mode_ == "sync") {
        for (auto &worker : workers_) {
            worker->load_params(params, weight_load_clone_);
        }
        return;
    }

    if (weight_load_mode_ == "async") {
        for (auto &worker : workers_) {
            worker->load_params_async(params, weight_load_clone_);
        }
        for (auto &worker : workers_) {
            worker->wait();
        }
        return;
    }

    const size_t group_size = std::max<size_t>(1, std::min(weight_load_group_size_, workers_.size()));
    for (size_t group_start = 0; group_start < workers_.size(); group_start += group_size) {
        const size_t group_end = std::min(group_start + group_size, workers_.size());
        for (size_t i = group_start; i < group_end; ++i) {
            workers_[i]->load_params_async(params, weight_load_clone_);
        }
        for (size_t i = group_start; i < group_end; ++i) {
            workers_[i]->wait();
        }
    }
}

//------------------------------------------------------
// load_param
//------------------------------------------------------
void InferEngine::process_weights_after_loading() {
    // Process the weights after loading on all workers
    for (auto &worker : workers_) {
        worker->process_weights_after_loading();
    }
}

//------------------------------------------------------
// state_dict
//------------------------------------------------------
std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> InferEngine::state_dict() {
    std::vector<std::unordered_map<std::string, infinicore::nn::Parameter>> results;
    if (0 == workers_.size()) {
        throw std::runtime_error(" Model object not found. ");
    }

    for (auto &worker : workers_) {
        results.push_back(worker->state_dict());
    }
    return results;
}

//------------------------------------------------------
// forward
//------------------------------------------------------
infinilm::InfinilmModel::Input
InferEngine::Input::to_model_input(infinicore::Device device) const {

    auto to_device = [&](const std::optional<infinicore::Tensor> &t)
        -> std::optional<infinicore::Tensor> {
        return t.has_value() ? t.value()->to(device) : t;
    };
    auto to_device_vec = [&](const std::optional<std::vector<infinicore::Tensor>> &vec)
        -> std::optional<std::vector<infinicore::Tensor>> {
        if (!vec.has_value()) {
            return vec;
        }
        std::vector<infinicore::Tensor> result;
        result.reserve(vec->size());
        for (const auto &t : vec.value()) {
            result.push_back(t->to(device));
        }
        return result;
    };

    infinilm::InfinilmModel::Input input = {
        to_device(input_ids), // @todo: on device in the future
        to_device(position_ids),
        to_device(past_sequence_lengths), // @todo: on device in the future
        to_device(total_sequence_lengths),
        to_device(input_offsets),
        to_device(cu_seqlens),
        to_device(block_tables),
        to_device(slot_mapping),
        to_device_vec(pixel_values),
        to_device_vec(image_bound),
        to_device_vec(tgt_sizes),
    };

    infinilm::global_state::get_forward_context().attn_metadata = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping};

    global_state::get_forward_context().mm_metadata = {
        image_req_ids};

    return input;
}

InferEngine::Output InferEngine::forward(const InferEngine::Input &input) {
    // Trigger each worker to run inference
    for (auto &worker : workers_) {
        worker->run(input);
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }

    return workers_[0]->get_output();
}

void InferEngine::compile() {
    for (auto &worker : workers_) {
        worker->compile();
    }
    // Wait for all workers
    for (auto &worker : workers_) {
        worker->wait();
    }
}

//------------------------------------------------------
// Destructor
//------------------------------------------------------
InferEngine::~InferEngine() {
    // Close all workers
    for (auto &worker : workers_) {
        worker->close();
    }
}

const distributed::DistConfig &InferEngine::get_dist_config() const {
    return communication_group_.get_dist_config();
}

//------------------------------------------------------
// reset_cache (overloaded with CacheConfig)
//------------------------------------------------------
void InferEngine::reset_cache(const cache::CacheConfig *new_config) {
    for (auto &worker : workers_) {
        worker->reset_cache(new_config);
    }
    for (auto &worker : workers_) {
        worker->wait();
    }
    cache_config_ = new_config->unique_copy();
    this->compile();
}

} // namespace infinilm::engine
