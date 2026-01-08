#include "infer_engine.hpp"
#include "spdlog/spdlog.h"

namespace infinilm::engine {

//------------------------------------------------------
// Constructor
//------------------------------------------------------
InferEngine::InferEngine(
    const InfinilmModel::Config &config,
    const distributed::DistConfig &distributed_config,
    infinicore::Device::Type device_type,
    const cache::CacheConfig *cache_config) // Changed parameter
    : communication_group_(distributed_config, device_type),
      model_config_(config) {

    if (cache_config != nullptr) {
        cache_config_ = cache_config->unique_copy();
    }
    // Create one RankWorker per rank
    int world_size = communication_group_.get_world_size();
    workers_.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
        workers_.emplace_back(std::make_unique<RankWorker>(
            model_config_,
            communication_group_.get_rank_info(r),
            cache_config_ != nullptr ? cache_config_.get() : nullptr));
    }
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

    return {
        input_ids, // @todo: on device in the future
        to_device(position_ids),
        past_sequence_lengths, // @todo: on device in the future
        to_device(total_sequence_lengths),
        to_device(input_offsets),
        to_device(block_tables),
        to_device(slot_mapping),
    };
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
}

} // namespace infinilm::engine
