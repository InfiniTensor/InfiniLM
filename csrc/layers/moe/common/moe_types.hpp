#pragma once

#include "topk_output.hpp"

#include "infinicore/tensor.hpp"

#include <cstddef>

namespace infinilm::layers::moe {

enum class DispatchOutputFormat {
    Standard,
    DeepEPNormal,
    DeepEPLL,
};

enum class CombineInputFormat {
    Standard,
    DeepEPNormal,
    DeepEPLL,
};

struct DispatchOutput {
    DispatchOutputFormat format = DispatchOutputFormat::Standard;
    infinicore::Tensor hidden_states;
    infinicore::Tensor hidden_states_scale;
    TopKOutput topk_output;
    infinicore::Tensor expert_map;
};

struct MoeRoutingMetadata {
    infinicore::Tensor sorted_token_ids;
    infinicore::Tensor expert_ids;
    infinicore::Tensor num_tokens_post_padded;

    infinicore::Tensor expert_offsets;
    infinicore::Tensor blockscale_offsets;
    infinicore::Tensor problem_sizes1;
    infinicore::Tensor problem_sizes2;
    infinicore::Tensor input_permutation;
    infinicore::Tensor output_permutation;

    bool has_grouped_gemm_metadata = false;
};

struct CombineInput {
    CombineInputFormat format = CombineInputFormat::Standard;
    infinicore::Tensor hidden_states;
    TopKOutput topk_output;
    MoeRoutingMetadata routing_metadata;
};

struct MoeWeights {
    infinicore::Tensor packed_w13;
    infinicore::Tensor packed_w2;

    bool empty() const {
        return !packed_w13 && !packed_w2;
    }

    bool has_packed_dense_weights() const {
        return packed_w13 && packed_w2;
    }
};

struct MoeWorkspace {
    infinicore::Tensor ep_gathered_hidden_states;
    infinicore::Tensor ep_gathered_topk_weights;
    infinicore::Tensor ep_gathered_topk_ids;
    infinicore::Tensor ep_reduced_hidden_states;
    infinicore::Tensor fused_moe_output;

    infinicore::Tensor sorted_token_ids;
    infinicore::Tensor expert_ids;
    infinicore::Tensor num_tokens_post_padded;
    infinicore::Tensor expert_offsets;
    infinicore::Tensor blockscale_offsets;
    infinicore::Tensor problem_sizes1;
    infinicore::Tensor problem_sizes2;
    infinicore::Tensor input_permutation;
    infinicore::Tensor output_permutation;

    size_t sorted_token_ids_capacity = 0;
    size_t expert_ids_capacity = 0;
    size_t ep_gathered_tokens_capacity = 0;
    size_t ep_reduced_tokens_capacity = 0;
    size_t fused_moe_output_tokens_capacity = 0;
    size_t blockscale_offsets_capacity = 0;
    size_t permutation_capacity = 0;
    size_t prepared_num_experts = 0;
};

} // namespace infinilm::layers::moe
