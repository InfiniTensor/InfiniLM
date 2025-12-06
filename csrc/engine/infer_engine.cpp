#include "infer_engine.hpp"
#include "spdlog/spdlog.h"

namespace infinilm::engine {

//------------------------------------------------------
// Constructor
//------------------------------------------------------
InferEngine::InferEngine(
    const std::any &config,
    const distributed::DistConfig &distributed_config,
    infinicore::Device::Type device_type)
    : communication_group_(distributed_config, device_type),
      model_config_(config) {
    spdlog::info("Launch InferEngine with {}", std::string(distributed_config));
    // Create one RankWorker per rank
    int world_size = communication_group_.get_world_size();
    workers_.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
        workers_.emplace_back(std::make_unique<RankWorker>(model_config_, communication_group_.get_rank_info(r)));
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
std::unordered_map<std::string, infinicore::nn::Parameter> InferEngine::state_dict() {
    if (0 == workers_.size()) {
        throw std::runtime_error(" Model object not found. ");
    }
    return workers_[0]->state_dict();
}

//------------------------------------------------------
// generate
//------------------------------------------------------
infinicore::Tensor InferEngine::generate(const infinicore::Tensor &input_ids,
                                         const infinicore::Tensor &position_ids) {
    // Trigger each worker to run inference
    for (auto &worker : workers_) {
        worker->run(std::vector<std::any>({input_ids, position_ids}));
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

} // namespace infinilm::engine
