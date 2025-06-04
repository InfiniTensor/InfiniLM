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

        CHECK_DTYPE_ANY_INT(dt_i);
        CHECK_DTYPE(dt_p, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        CHECK_OR_RETURN(result_desc->ndim() == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(probs_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(probs_desc->stride(0) == 1, INFINI_STATUS_BAD_TENSOR_STRIDES);

        return utils::Result<RandomSampleInfo>({dt_i, dt_p, probs_desc->dim(0)});
    }
};

} // namespace op::random_sample

#endif // __RANDOM_SAMPLE_INFO_H__
