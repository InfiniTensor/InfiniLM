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
    const cache::CacheConfig &cache_config) // Changed parameter
    : communication_group_(distributed_config, device_type),
      model_config_(config),
      cache_config_(cache_config) {

    spdlog::info("Launch InferEngine with {}", std::string(distributed_config));
    spdlog::info("Cache configuration: type={}, layers={}, max_kv_cache_length={}",
                 static_cast<int>(cache_config_.type),
                 cache_config_.num_layers,
                 cache_config_.max_kv_cache_length);

    // Try to extract model configuration to override default cache parameters if needed
    try {
        if (config.type() == typeid(models::llama::LlamaConfig)) {
            const auto &llama_config = std::any_cast<models::llama::LlamaConfig>(config);

            cache_config_.num_layers = llama_config.num_hidden_layers;
            cache_config_.max_kv_cache_length = llama_config.max_position_embeddings;

            spdlog::info("Updated cache config from model: layers={}, max_kv_cache_length={}",
                         cache_config_.num_layers, cache_config_.max_kv_cache_length);
        }
    } catch (...) {
        spdlog::warn("Could not extract model config, using provided CacheConfig");
    }

    // Create CacheManager with one cache per worker
    create_cache_manager();
}

//------------------------------------------------------
// Helper method to create cache manager
//------------------------------------------------------
void InferEngine::create_cache_manager() {
    int world_size = communication_group_.get_world_size();

    spdlog::info("Creating CacheManager with config: type={}, layers={}, initial_capacity={}, growth_factor={}",
                 static_cast<int>(cache_config_.type),
                 cache_config_.num_layers,
                 cache_config_.initial_capacity,
                 cache_config_.growth_factor);

    // Create CacheManager using CacheConfig
    cache_manager_ = std::make_unique<cache::CacheManager>(world_size, cache_config_);

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
void InferEngine::reset_cache(size_t pos) {
    cache_manager_->reset_pos(pos);
}

//------------------------------------------------------
// reset_cache (overloaded with CacheConfig)
//------------------------------------------------------
void InferEngine::reset_cache(const cache::CacheConfig &new_config, size_t pos) {
    for (size_t i = 0; i < workers_.size(); ++i) {
        workers_[i]->reset_cache(new_config, pos);
    }
}

} // namespace infinilm::engine
