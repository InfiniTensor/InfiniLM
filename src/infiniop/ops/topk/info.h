#ifndef __TOPK_INFO_H__
#define __TOPK_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::topk {

enum TopKStrategy {
    DEEPSEEK_V3,
    STANDARD_SOFTMAX
};

class TopKInfo {
    TopKInfo() = default;

  public:
    int num_tokens, num_experts, k;
    infiniDtype_t data_type;
    size_t workspace_size;
    TopKStrategy strategy;
    // New parameters for DeepseekV3
    int n_group;
    int topk_group;

    static utils::Result<TopKInfo>
    create(infiniopTensorDescriptor_t input_desc,
           infiniopTensorDescriptor_t output_val_desc,
           infiniopTensorDescriptor_t output_ind_desc,
           infiniopTensorDescriptor_t bias_desc, int k_val,
           TopKStrategy strategy, int n_group, int topk_group) {
        if (input_desc->ndim() != 2 || output_val_desc->ndim() != 2 ||
            output_ind_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        int num_tokens = input_desc->shape()[0];
        int num_experts = input_desc->shape()[1];
        if (output_val_desc->shape()[0] != static_cast<size_t>(num_tokens) ||
            output_val_desc->shape()[1] != static_cast<size_t>(k_val) ||
            output_ind_desc->shape()[0] != static_cast<size_t>(num_tokens) ||
            output_ind_desc->shape()[1] != static_cast<size_t>(k_val)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (bias_desc) {
            if (bias_desc->ndim() != 1u ||
                bias_desc->shape()[0] != static_cast<size_t>(num_experts)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
        size_t workspace_size = 0;
        if (strategy == DEEPSEEK_V3) {
            workspace_size =
                num_tokens * num_experts * sizeof(float) +
                num_tokens * n_group * sizeof(float) +
                num_tokens * topk_group * sizeof(int) +
                num_tokens * num_experts * sizeof(char);
        } else { // STANDARD_SOFTMAX
            workspace_size = num_tokens * num_experts * sizeof(input_desc->dtype());
        }

        return utils::Result<TopKInfo>(
            TopKInfo{num_tokens,   num_experts, k_val,
                     input_desc->dtype(),  workspace_size, strategy, n_group,
                     topk_group});
    }
};

} // namespace op::topk

#endif // __TOPK_INFO_H__ 