#include "global_state.hpp"
#include "../utils.hpp"
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

} // namespace infinilm::global_state
