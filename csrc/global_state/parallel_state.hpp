#pragma once

#include "../engine/distributed/distributed.hpp"

namespace infinilm::global_state {

void initialize_model_parallel(const engine::distributed::RankInfo &rank_info);

/**
 * @brief get the world size of the tensor model parallel group.
 */
const size_t get_tensor_model_parallel_world_size();

/**
 * @brief get the rank of the current process in the tensor model parallel group.
 */
const size_t get_tensor_model_parallel_rank();

/**
 * @brief get the rank_info of the current process in the tensor model parallel group.
 */
const engine::distributed::RankInfo &get_tensor_model_parallel_rank_info();

} // namespace infinilm::global_state
