#ifndef __MOE_DISPATCH_INFO_H__
#define __MOE_DISPATCH_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::moe_dispatch {

class MoEDispatchInfo {
    MoEDispatchInfo() = default;

  public:
    int num_tokens, k, hidden_dim, num_experts;
    infiniDtype_t data_type;
    infiniDtype_t index_type;

    static utils::Result<MoEDispatchInfo>
    create(infiniopTensorDescriptor_t input_desc,
           infiniopTensorDescriptor_t indices_desc,
           infiniopTensorDescriptor_t permuted_output_desc,
           infiniopTensorDescriptor_t aux_info_desc, int num_experts) {

        auto data_type = input_desc->dtype();
        if (permuted_output_desc->dtype() != data_type) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (data_type != INFINI_DTYPE_F16 && data_type != INFINI_DTYPE_BF16 &&
            data_type != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto index_type = indices_desc->dtype();
        if (index_type != INFINI_DTYPE_I32 ||
            aux_info_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (input_desc->ndim() != 2 || indices_desc->ndim() != 2 ||
            permuted_output_desc->ndim() != 2 || aux_info_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        int num_tokens = input_desc->shape()[0];
        int hidden_dim = input_desc->shape()[1];
        int k = indices_desc->shape()[1];

        if (indices_desc->shape()[0] != (size_t)num_tokens) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (permuted_output_desc->shape()[0] != (size_t)num_tokens * k ||
            permuted_output_desc->shape()[1] != (size_t)hidden_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (aux_info_desc->shape()[0] != (size_t)num_tokens * k ||
            aux_info_desc->shape()[1] != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->stride(1) != 1 ||
            permuted_output_desc->stride(1) != 1 ||
            indices_desc->stride(1) != 1 || aux_info_desc->stride(1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<MoEDispatchInfo>(MoEDispatchInfo{
            num_tokens, k, hidden_dim, num_experts, data_type, index_type});
    }
};

} // namespace op::moe_dispatch

#endif // __MOE_DISPATCH_INFO_H__ 