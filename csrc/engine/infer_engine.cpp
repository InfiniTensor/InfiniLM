#include "infer_engine.hpp"
#include "../models/llama/llama_config.hpp"
#include "spdlog/spdlog.h"

namespace infinilm::engine {

//------------------------------------------------------
// Constructor
//------------------------------------------------------
InferEngine::InferEngine(
    const std::any &config,
    const distributed::DistConfig &distributed_config,
    infinicore::Device::Type device_type,
    cache::CacheType cache_type)
    : communication_group_(distributed_config, device_type),
      model_config_(config) {

    spdlog::info("Launch InferEngine with {}", std::string(distributed_config));

    // Determine number of layers from config if available
    size_t num_layers = 32;                // Default value, should extract from config
    size_t max_position_embeddings = 4096; // Default value

    // Try to extract model configuration for cache parameters
    try {
        // Assuming config contains LlamaConfig or similar
        // This needs to be adapted based on actual config type
        if (config.type() == typeid(models::llama::LlamaConfig)) {
            const auto &llama_config = std::any_cast<models::llama::LlamaConfig>(config);
            num_layers = llama_config.num_hidden_layers;
            max_position_embeddings = llama_config.max_position_embeddings;
        }
    } catch (...) {
        spdlog::warn("Could not extract model config for cache parameters, using defaults");
    }

    // Create CacheManager with one cache per worker
    int world_size = communication_group_.get_world_size();
    cache_manager_ = std::make_unique<cache::CacheManager>(
        world_size, cache_type, num_layers, max_position_embeddings);

    // Create one RankWorker per rank, passing cache pointer
    workers_.reserve(world_size);
    for (int r = 0; r < world_size; ++r) {
        workers_.emplace_back(std::make_unique<RankWorker>(
            model_config_,
            communication_group_.get_rank_info(r),
            cache_manager_->get_raw_cache_ptr(r)));
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

//------------------------------------------------------
// reset_cache
//------------------------------------------------------
void InferEngine::reset_cache(size_t pos, bool async) {
    if (!async) {
        cache_manager_->reset_all(pos);
    } else {
        // Asynchronous reset: reset each worker individually
        for (size_t i = 0; i < workers_.size(); ++i) {
            workers_[i]->reset_cache(pos, true);
        }
    }
}

} // namespace infinilm::engine
