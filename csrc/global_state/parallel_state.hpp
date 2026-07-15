#pragma once

#include "../engine/distributed/distributed.hpp"
#include <cstddef>

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

struct PipelineLayerRange {
    size_t start;
    size_t end;
};

const size_t get_pipeline_model_parallel_world_size();

const size_t get_pipeline_model_parallel_rank();

bool is_first_pipeline_stage();

bool is_last_pipeline_stage();

PipelineLayerRange get_pipeline_layer_range(size_t num_layers);

} // namespace infinilm::global_state
