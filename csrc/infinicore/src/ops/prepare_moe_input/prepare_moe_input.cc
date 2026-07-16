#include "infinicore/ops/prepare_moe_input.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(PrepareMoeInput);

PrepareMoeInput::PrepareMoeInput(Tensor expert_offsets,
                                 Tensor blockscale_offsets,
                                 Tensor problem_sizes1,
                                 Tensor problem_sizes2,
                                 Tensor input_permutation,
                                 Tensor output_permutation,
                                 const Tensor &topk_ids,
                                 const size_t num_experts,
                                 const size_t n,
                                 const size_t k) {
    if (blockscale_offsets) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
            expert_offsets, blockscale_offsets, problem_sizes1, problem_sizes2, input_permutation, output_permutation, topk_ids);
    } else {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
            expert_offsets, problem_sizes1, problem_sizes2, input_permutation, output_permutation, topk_ids);
    }
    INFINICORE_GRAPH_OP_DISPATCH(
        expert_offsets->device().type(),
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        topk_ids,
        num_experts,
        n,
        k);
}

void PrepareMoeInput::execute(Tensor expert_offsets,
                              Tensor blockscale_offsets,
                              Tensor problem_sizes1,
                              Tensor problem_sizes2,
                              Tensor input_permutation,
                              Tensor output_permutation,
                              const Tensor &topk_ids,
                              const size_t num_experts,
                              const size_t n,
                              const size_t k) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        PrepareMoeInput,
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        topk_ids,
        num_experts,
        n,
        k);
}

PrepareMoeInputOutput prepare_moe_input(
    const Tensor &topk_ids,
    size_t num_experts,
    size_t n,
    size_t k) {
    auto shape = topk_ids->shape();
    INFINICORE_ASSERT(shape.size() == 2);
    const size_t topk_length = shape[0] * shape[1];

    auto expert_offsets = Tensor::empty({num_experts + 1}, DataType::kInt32, topk_ids->device());
    auto problem_sizes1 = Tensor::empty({num_experts, 3}, DataType::kInt32, topk_ids->device());
    auto problem_sizes2 = Tensor::empty({num_experts, 3}, DataType::kInt32, topk_ids->device());
    auto input_permutation = Tensor::empty({topk_length}, DataType::kInt32, topk_ids->device());
    auto output_permutation = Tensor::empty({topk_length}, DataType::kInt32, topk_ids->device());

    prepare_moe_input_(
        expert_offsets,
        Tensor(),
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        topk_ids,
        num_experts,
        n,
        k);

    return PrepareMoeInputOutput{
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
    };
}

void prepare_moe_input_(
    Tensor expert_offsets,
    Tensor blockscale_offsets,
    Tensor problem_sizes1,
    Tensor problem_sizes2,
    Tensor input_permutation,
    Tensor output_permutation,
    const Tensor &topk_ids,
    size_t num_experts,
    size_t n,
    size_t k) {
    PrepareMoeInput::execute(
        expert_offsets,
        blockscale_offsets,
        problem_sizes1,
        problem_sizes2,
        input_permutation,
        output_permutation,
        topk_ids,
        num_experts,
        n,
        k);
}

} // namespace infinicore::op
