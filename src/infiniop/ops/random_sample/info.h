#ifndef __RANDOM_SAMPLE_INFO_H__
#define __RANDOM_SAMPLE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::random_sample {

struct RandomSampleInfo {
    infiniDtype_t dt_i, dt_p;
    size_t n;

    static utils::Result<RandomSampleInfo> create(
        infiniopTensorDescriptor_t result_desc,
        infiniopTensorDescriptor_t probs_desc) {

        auto dt_i = result_desc->dtype();
        auto dt_p = probs_desc->dtype();

        CHECK_DTYPE(dt_i,
                    INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                    INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64);
        CHECK_DTYPE(dt_p, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        CHECK_API_OR(result_desc->ndim(), 0,
                     return INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_API_OR(probs_desc->ndim(), 1,
                     return INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_API_OR(probs_desc->stride(0), 1,
                     return INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<RandomSampleInfo>({dt_i, dt_p, probs_desc->dim(0)});
    }
};

} // namespace op::random_sample

#endif // __RANDOM_SAMPLE_INFO_H__
