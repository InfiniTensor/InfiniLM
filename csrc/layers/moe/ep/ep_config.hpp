#pragma once

#include <cstddef>

namespace infinilm::layers::moe {

enum class EPBackend {
    Disabled,
    AllGatherReduceScatter,
    LocalAllReduce,
    DeepEP,
};

enum class DeepEPMode {
    Normal,
    LowLatency,
    Auto,
};

struct EPConfig {
    EPBackend backend = EPBackend::Disabled;
    size_t ep_rank = 0;
    size_t ep_size = 1;
    DeepEPMode deepep_mode = DeepEPMode::Auto;
};

struct ExpertPlacement {
    size_t global_num_experts = 0;
    size_t local_num_experts = 0;
    size_t local_expert_start = 0;
    size_t local_expert_end = 0;

    bool owns(size_t global_expert) const {
        return global_expert >= local_expert_start && global_expert < local_expert_end;
    }

    size_t global_to_local(size_t global_expert) const {
        return global_expert - local_expert_start;
    }
};

const char *ep_backend_name(EPBackend backend);

EPConfig make_ep_config();

ExpertPlacement make_expert_placement(const EPConfig &config,
                                      size_t global_num_experts);

} // namespace infinilm::layers::moe
