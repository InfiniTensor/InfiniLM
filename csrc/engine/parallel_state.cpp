#include "parallel_state.hpp"

namespace infinilm::engine {

namespace {

thread_local std::shared_ptr<engine::distributed::RankInfo> _TP;

} // namespace

void initialize_model_parallel(const engine::distributed::RankInfo &rank_info) {
    if (_TP == nullptr) {
        _TP = std::make_shared<engine::distributed::RankInfo>();
    }
    _TP->device = rank_info.device;
    _TP->tp_size = rank_info.tp_size;
    _TP->tp_rank = rank_info.tp_rank;
    _TP->comm = rank_info.comm;
}

// Return world size for the tensor model parallel group.
const size_t get_tensor_model_parallel_world_size() {
    assert(_TP != nullptr);
    return _TP->tp_size;
}

// Return my rank for the tensor model parallel group.
const size_t get_tensor_model_parallel_rank() {
    assert(_TP != nullptr);
    return _TP->tp_rank;
}

// Return rank_info.
const engine::distributed::RankInfo &get_tensor_model_parallel_rank_info() {
    assert(_TP != nullptr);
    return *_TP;
}

} // namespace infinilm::engine
