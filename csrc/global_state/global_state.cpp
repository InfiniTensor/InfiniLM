#include "global_state.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>

namespace infinilm::global_state {

namespace {

thread_local ForwardContext *_forward_context{nullptr};

thread_local InfinilmConfig *_infinilm_config{nullptr};

thread_local const engine::distributed::RankInfo *_TP{nullptr};

} // namespace
} // namespace infinilm::global_state

namespace infinilm::global_state {

void initialize_forward_context(ForwardContext &forward_context) {
    ASSERT(nullptr == _forward_context && "Forward context is already initialized, cannot be initialized again.");
    _forward_context = &forward_context;
}

ForwardContext &get_forward_context() {
    return *_forward_context;
}

} // namespace infinilm::global_state

namespace infinilm::global_state {

void initialize_infinilm_config(const std::shared_ptr<InfinilmConfig> &config) {
    ASSERT(nullptr == _infinilm_config);
    ASSERT(nullptr != config);
    _infinilm_config = config.get();
}

const InfinilmConfig &get_infinilm_config() {
    ASSERT(nullptr != _infinilm_config && "Current Infinilm config is not set.");
    return *_infinilm_config;
}

} // namespace infinilm::global_state

namespace infinilm::global_state {

void initialize_model_parallel(const engine::distributed::RankInfo &rank_info) {
    ASSERT(nullptr == _TP && "Tensor model parallel state is already initialized, cannot be initialized again.");
    _TP = &rank_info;
}

const size_t get_tensor_model_parallel_world_size() {
    ASSERT(nullptr != _TP && "Tensor model parallel state is not initialized.");
    return _TP->tp_size;
}

const size_t get_tensor_model_parallel_rank() {
    ASSERT(nullptr != _TP && "Tensor model parallel state is not initialized.");
    return _TP->tp_rank;
}

const engine::distributed::RankInfo &get_tensor_model_parallel_rank_info() {
    ASSERT(nullptr != _TP && "Tensor model parallel state is not initialized.");
    return *_TP;
}

const size_t get_pipeline_model_parallel_world_size() {
    ASSERT(nullptr != _TP && "Pipeline model parallel state is not initialized.");
    return _TP->pp_size;
}

const size_t get_pipeline_model_parallel_rank() {
    ASSERT(nullptr != _TP && "Pipeline model parallel state is not initialized.");
    return _TP->pp_rank;
}

bool is_first_pipeline_stage() {
    ASSERT(nullptr != _TP && "Pipeline model parallel state is not initialized.");
    return _TP->is_first_pipeline_stage();
}

bool is_last_pipeline_stage() {
    ASSERT(nullptr != _TP && "Pipeline model parallel state is not initialized.");
    return _TP->is_last_pipeline_stage();
}

PipelineLayerRange get_pipeline_layer_range(size_t num_layers) {
    const size_t pp_size = get_pipeline_model_parallel_world_size();
    const size_t pp_rank = get_pipeline_model_parallel_rank();
    if (pp_size == 0 || pp_rank >= pp_size) {
        throw std::runtime_error("Invalid pipeline parallel state");
    }
    if (num_layers < pp_size) {
        throw std::invalid_argument("Pipeline parallel size must not exceed the number of model layers");
    }

    const size_t base = num_layers / pp_size;
    const size_t remainder = num_layers % pp_size;
    const size_t start = pp_rank * base + std::min(pp_rank, remainder);
    const size_t count = base + (pp_rank < remainder ? 1 : 0);
    return {start, start + count};
}

} // namespace infinilm::global_state
