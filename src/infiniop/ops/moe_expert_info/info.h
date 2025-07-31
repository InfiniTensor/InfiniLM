#ifndef __MOE_EXPERT_INFO_INFO_H__
#define __MOE_EXPERT_INFO_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::moe_expert_info {

/**
 * @brief Holds metadata and performs validation for the MoEExpertInfo operator.
 *
 * This class checks the tensor descriptors for the operation that calculates
 * expert token counts and offsets. It ensures that the data types, dimensions,
 * and shapes are consistent and valid before proceeding with the computation.
 */
class MoEExpertInfoInfo {
    // Private constructor to enforce creation via the static factory method.
    MoEExpertInfoInfo() = default;

  public:
    int num_tokens, k, num_experts;
    infiniDtype_t index_type;

    /**
     * @brief Factory method to create and validate an MoEExpertInfoInfo instance.
     *
     * @param topk_ind_desc Descriptor for the input tensor containing expert indices from the TopK operation.
     * @param expert_counts_desc Descriptor for the output tensor that will store the token count for each expert.
     * @param expert_offsets_desc Descriptor for the output tensor that will store the token offset for each expert.
     * @return A Result object containing either a valid MoEExpertInfoInfo instance or an error status.
     */
    static utils::Result<MoEExpertInfoInfo>
    create(infiniopTensorDescriptor_t topk_ind_desc,
           infiniopTensorDescriptor_t expert_counts_desc,
           infiniopTensorDescriptor_t expert_offsets_desc) {

        // 1. Validate Data Types
        auto index_type = topk_ind_desc->dtype();
        if (index_type != INFINI_DTYPE_I32 ||
            expert_counts_desc->dtype() != INFINI_DTYPE_I32 ||
            expert_offsets_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 2. Validate Dimensions
        if (topk_ind_desc->ndim() != 2 || expert_counts_desc->ndim() != 1 ||
            expert_offsets_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 3. Validate Shapes
        int num_tokens = topk_ind_desc->shape()[0];
        int k = topk_ind_desc->shape()[1];
        int num_experts = expert_counts_desc->shape()[0];

        if (expert_offsets_desc->shape()[0] != (size_t)num_experts) {
            // The number of experts derived from counts and offsets tensors must match.
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 4. Validate Strides (check for contiguity)
        if (topk_ind_desc->stride(1) != 1) {
            // The innermost dimension must be contiguous.
            // 1D tensors (counts, offsets) are always contiguous.
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        // 5. Return a valid info object on success
        return utils::Result<MoEExpertInfoInfo>(
            MoEExpertInfoInfo{num_tokens, k, num_experts, index_type});
    }
};

} // namespace op::moe_expert_info

#endif // __MOE_EXPERT_INFO_INFO_H__