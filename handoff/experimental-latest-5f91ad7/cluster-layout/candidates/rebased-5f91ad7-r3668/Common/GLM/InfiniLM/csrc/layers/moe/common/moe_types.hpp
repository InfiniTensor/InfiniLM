#pragma once

#include "topk_output.hpp"

#include "runtime/tensor.hpp"

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
    infinilm::runtime::Tensor hidden_states;
    infinilm::runtime::Tensor hidden_states_scale;
    TopKOutput topk_output;
    infinilm::runtime::Tensor expert_map;
};

struct MoeRoutingMetadata {
    infinilm::runtime::Tensor sorted_token_ids;
    infinilm::runtime::Tensor expert_ids;
    infinilm::runtime::Tensor num_tokens_post_padded;

    infinilm::runtime::Tensor expert_offsets;
    infinilm::runtime::Tensor blockscale_offsets;
    infinilm::runtime::Tensor problem_sizes1;
    infinilm::runtime::Tensor problem_sizes2;
    infinilm::runtime::Tensor input_permutation;
    infinilm::runtime::Tensor output_permutation;

    bool has_grouped_gemm_metadata = false;
};

struct CombineInput {
    CombineInputFormat format = CombineInputFormat::Standard;
    infinilm::runtime::Tensor hidden_states;
    TopKOutput topk_output;
    MoeRoutingMetadata routing_metadata;
};

struct MoeWeights {
    infinilm::runtime::Tensor packed_w13;
    infinilm::runtime::Tensor packed_w2;
    infinilm::runtime::Tensor packed_b13{};
    infinilm::runtime::Tensor packed_b2{};

    bool empty() const {
        return !packed_w13 && !packed_w2;
    }

    bool has_packed_dense_weights() const {
        return packed_w13 && packed_w2;
    }
};

struct MoeWorkspace {
    infinilm::runtime::Tensor ep_gathered_hidden_states;
    infinilm::runtime::Tensor ep_gathered_topk_weights;
    infinilm::runtime::Tensor ep_gathered_topk_ids;
    infinilm::runtime::Tensor ep_reduced_hidden_states;
    infinilm::runtime::Tensor fused_moe_output;

    infinilm::runtime::Tensor sorted_token_ids;
    infinilm::runtime::Tensor expert_ids;
    infinilm::runtime::Tensor num_tokens_post_padded;
    infinilm::runtime::Tensor expert_offsets;
    infinilm::runtime::Tensor blockscale_offsets;
    infinilm::runtime::Tensor problem_sizes1;
    infinilm::runtime::Tensor problem_sizes2;
    infinilm::runtime::Tensor input_permutation;
    infinilm::runtime::Tensor output_permutation;

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
