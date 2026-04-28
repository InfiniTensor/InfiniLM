#ifndef __PER_TENSOR_QUANT_INT8_INFO_H__
#define __PER_TENSOR_QUANT_INT8_INFO_H__

#include "../../../../utils.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::per_tensor_quant_int8 {

class PerTensorQuantI8Info {
private:
    PerTensorQuantI8Info() = default;

public:
    infiniDtype_t dtype, packed_type;
    size_t batch_size, channel, hidden_dim, width;
    ptrdiff_t strides_0, strides_1, strides_2, strides_3;
    ptrdiff_t p_strides_0, p_strides_1, p_strides_2, p_strides_3;
    int num_elements;
    bool is_static;

    static utils::Result<PerTensorQuantI8Info> createPerTensorQuantI8Info(
        infiniopTensorDescriptor_t x_packed_desc,
        infiniopTensorDescriptor_t x_scale_desc,
        infiniopTensorDescriptor_t x_zero_desc,
        infiniopTensorDescriptor_t x_desc) {

        CHECK_OR_RETURN(
            x_packed_desc != nullptr && x_scale_desc != nullptr && x_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t dtype = x_desc->dtype();
        const infiniDtype_t packed_type = x_packed_desc->dtype();

        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE(packed_type, INFINI_DTYPE_I8);

        auto shape = x_desc->shape();
        CHECK_SAME_SHAPE(shape, x_packed_desc->shape());

        auto ndim = x_desc->ndim();
        CHECK_OR_RETURN(ndim <= 4,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t width = shape[ndim - 1];
        size_t hidden_dim = (ndim > 1 ? shape[ndim - 2] : 1);
        size_t channel = (ndim > 2 ? shape[ndim - 3] : 1);
        size_t batch_size = (ndim > 3 ? shape[ndim - 4] : 1);

        ptrdiff_t strides_3 = x_desc->strides()[ndim - 1];
        ptrdiff_t strides_2 = (ndim > 1 ? x_desc->strides()[ndim - 2] : 0);
        ptrdiff_t strides_1 = (ndim > 2 ? x_desc->strides()[ndim - 3] : 0);
        ptrdiff_t strides_0 = (ndim > 3 ? x_desc->strides()[ndim - 4] : 0);

        ptrdiff_t p_strides_3 = x_packed_desc->strides()[ndim - 1];
        ptrdiff_t p_strides_2 = (ndim > 1 ? x_packed_desc->strides()[ndim - 2] : 0);
        ptrdiff_t p_strides_1 = (ndim > 2 ? x_packed_desc->strides()[ndim - 3] : 0);
        ptrdiff_t p_strides_0 = (ndim > 3 ? x_packed_desc->strides()[ndim - 4] : 0);

        int num_elements = 1;
        for (int i = 0; i < (int)ndim; i++) {
            num_elements *= static_cast<int>(shape[i]);
        }

        return utils::Result<PerTensorQuantI8Info>(PerTensorQuantI8Info{
            dtype,
            packed_type,
            batch_size, channel, hidden_dim, width,
            strides_0, strides_1, strides_2, strides_3,
            p_strides_0, p_strides_1, p_strides_2, p_strides_3,
            num_elements});
    }
};

} // namespace op::per_tensor_quant_int8

#endif //  __PER_TENSOR_QUANT_INT8_INFO_H__
