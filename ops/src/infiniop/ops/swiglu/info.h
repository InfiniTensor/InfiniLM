#ifndef __SWIGLU_CUDA_INFO_H__
#define __SWIGLU_CUDA_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::swiglu_cuda {

class SwiGLUCudaInfo {
    SwiGLUCudaInfo() = default;

public:
    infiniDtype_t dtype;
    size_t length;
    size_t batch, seq_len, hidden_dim;
    ptrdiff_t c_strides_0, c_strides_1, c_strides_2;
    ptrdiff_t a_strides_0, a_strides_1, a_strides_2;
    ptrdiff_t b_strides_0, b_strides_1, b_strides_2;

    static utils::Result<SwiGLUCudaInfo> createSwiGLUCudaInfo(infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc) {
        auto dtype = c_desc->dtype();
        if (dtype != a_desc->dtype() || dtype != b_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

        auto shape = c_desc->shape();
        CHECK_SAME_SHAPE(shape, a_desc->shape(), b_desc->shape());

        auto ndim = c_desc->ndim();
        size_t hidden_dim = shape[ndim - 1];
        size_t seq_len = shape[ndim - 2];
        size_t batch = (ndim == 3 ? shape[0] : 1);

        size_t length = batch * seq_len * hidden_dim;

        ptrdiff_t c_strides_0 = (ndim == 3 ? c_desc->strides()[0] : 0);
        ptrdiff_t c_strides_1 = (ndim == 3 ? c_desc->strides()[1] : c_desc->strides()[0]);
        ptrdiff_t c_strides_2 = (ndim == 3 ? c_desc->strides()[2] : c_desc->strides()[1]);
        ptrdiff_t a_strides_0 = (ndim == 3 ? a_desc->strides()[0] : 0);
        ptrdiff_t a_strides_1 = (ndim == 3 ? a_desc->strides()[1] : a_desc->strides()[0]);
        ptrdiff_t a_strides_2 = (ndim == 3 ? a_desc->strides()[2] : a_desc->strides()[1]);
        ptrdiff_t b_strides_0 = (ndim == 3 ? b_desc->strides()[0] : 0);
        ptrdiff_t b_strides_1 = (ndim == 3 ? b_desc->strides()[1] : b_desc->strides()[0]);
        ptrdiff_t b_strides_2 = (ndim == 3 ? b_desc->strides()[2] : b_desc->strides()[1]);

        return utils::Result<SwiGLUCudaInfo>(SwiGLUCudaInfo{
            dtype,
            length,
            batch,
            seq_len,
            hidden_dim,
            c_strides_0,
            c_strides_1,
            c_strides_2,
            a_strides_0,
            a_strides_1,
            a_strides_2,
            b_strides_0,
            b_strides_1,
            b_strides_2});
    }
};

} // namespace op::swiglu_cuda

#endif // __SWIGLU_CUDA_INFO_H__
