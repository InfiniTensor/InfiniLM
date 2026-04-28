#ifndef __MASKED_SELECT__INFO_H__
#define __MASKED_SELECT__INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::masked_select {

class MaskedSelectInfo {
    MaskedSelectInfo() = default;

public:
    infiniDtype_t dtype;
    size_t ndim;
    size_t total_elements;

    std::vector<size_t> shape;

    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> mask_strides;

    static utils::Result<MaskedSelectInfo> create(infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t mask_desc) {
        auto dtype = input_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F32);
        CHECK_DTYPE(mask_desc->dtype(), INFINI_DTYPE_BOOL);

        auto shape = input_desc->shape();
        CHECK_SAME_SHAPE(shape, mask_desc->shape());

        auto ndim = input_desc->ndim();

        size_t total_elements = 1;
        for (auto &dim : shape) {
            total_elements *= dim;
        }

        auto input_strides = input_desc->strides();
        auto mask_strides = mask_desc->strides();

        return utils::Result<MaskedSelectInfo>(MaskedSelectInfo{
            dtype,
            ndim,
            total_elements,
            shape,
            input_strides,
            mask_strides});
    }
};

} // namespace op::masked_select

#endif // __MASKED_SELECT_INFO_H__
