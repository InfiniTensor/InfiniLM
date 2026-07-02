#pragma once

#include "base_runner.hpp"

namespace infinilm::layers::moe {

struct CudaFusedMoeRunnerInput {
    infinicore::Tensor hidden_states;
    TopKOutput topk_output;
    MoeRoutingMetadata routing_metadata;
};

struct CudaFusedMoeRunnerOutput {
    infinicore::Tensor hidden_states;
};

class CudaFusedMoeRunner final : public MoeRunnerCore {
public:
    CudaFusedMoeRunner(size_t num_local_experts,
                       size_t hidden_size,
                       size_t intermediate_size_per_partition,
                       size_t align_block_size);

    CombineInput run(const DispatchOutput &dispatch_output,
                     const MoeWeights &weights,
                     MoeWorkspace &workspace) const override;

private:
    CudaFusedMoeRunnerInput prepare_runner_input(const DispatchOutput &dispatch_output,
                                                 MoeWorkspace &workspace) const;

    CudaFusedMoeRunnerOutput run_fused_core(const CudaFusedMoeRunnerInput &runner_input,
                                            const MoeWeights &weights,
                                            MoeWorkspace &workspace) const;

    size_t num_local_experts_ = 0;
    size_t hidden_size_ = 0;
    size_t intermediate_size_per_partition_ = 0;
    size_t align_block_size_ = 16;
};

} // namespace infinilm::layers::moe
