#pragma once

#include "distributed/distributed.hpp"

namespace infinilm::engine {
/*
https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int | None = 1,
    backend: str | None = None,
) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.
        backend: name of torch distributed communication backend.
*/

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

} // namespace infinilm::engine
