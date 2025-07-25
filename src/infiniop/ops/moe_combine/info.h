#ifndef __MOE_COMBINE_INFO_H__
#define __MOE_COMBINE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::moe_combine {

class MoECombineInfo {
    MoECombineInfo() = default;

  public:
    int num_tokens, k, hidden_dim;
    infiniDtype_t data_type;
    infiniDtype_t index_type;

    static utils::Result<MoECombineInfo>
    create(infiniopTensorDescriptor_t permuted_input_desc,
           infiniopTensorDescriptor_t gating_weights_desc,
           infiniopTensorDescriptor_t aux_info_desc,
           infiniopTensorDescriptor_t output_desc) {

        auto data_type = permuted_input_desc->dtype();
        if (gating_weights_desc->dtype() != data_type ||
            output_desc->dtype() != data_type) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (data_type != INFINI_DTYPE_F16 && data_type != INFINI_DTYPE_BF16 &&
            data_type != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto index_type = aux_info_desc->dtype();
        if (index_type != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (permuted_input_desc->ndim() != 2 ||
            gating_weights_desc->ndim() != 2 || aux_info_desc->ndim() != 2 ||
            output_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        int num_tokens = output_desc->shape()[0];
        int hidden_dim = output_desc->shape()[1];
        int k = gating_weights_desc->shape()[1];

        if (gating_weights_desc->shape()[0] != (size_t)num_tokens) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (permuted_input_desc->shape()[0] != (size_t)num_tokens * k ||
            permuted_input_desc->shape()[1] != (size_t)hidden_dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (aux_info_desc->shape()[0] != (size_t)num_tokens * k ||
            aux_info_desc->shape()[1] != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (permuted_input_desc->stride(1) != 1 ||
            gating_weights_desc->stride(1) != 1 ||
            aux_info_desc->stride(1) != 1 || output_desc->stride(1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<MoECombineInfo>(MoECombineInfo{
            num_tokens, k, hidden_dim, data_type, index_type});
    }
};

} // namespace op::moe_combine

#endif // __MOE_COMBINE_INFO_H__ 