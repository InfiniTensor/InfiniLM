#include "parallel_state.hpp"

#include <cassert>

namespace infinilm::engine {

namespace {

thread_local std::shared_ptr<engine::distributed::RankInfo> _TP;

} // namespace

void initialize_model_parallel(const engine::distributed::RankInfo &rank_info) {
    assert(nullptr == _TP);

    _TP = std::make_shared<engine::distributed::RankInfo>();
    _TP->device = rank_info.device;
    _TP->tp_size = rank_info.tp_size;
    _TP->tp_rank = rank_info.tp_rank;
    _TP->comm = rank_info.comm;
}

const size_t get_tensor_model_parallel_world_size() {
    assert(nullptr != _TP);
    return _TP->tp_size;
}

const size_t get_tensor_model_parallel_rank() {
    assert(nullptr != _TP);
    return _TP->tp_rank;
}

const engine::distributed::RankInfo &get_tensor_model_parallel_rank_info() {
    assert(nullptr != _TP);
    return *_TP;
}

} // namespace infinilm::engine
