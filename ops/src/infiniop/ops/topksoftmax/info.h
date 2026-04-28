#ifndef __TOPKSOFTMAX_INFO_H__
#define __TOPKSOFTMAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::topksoftmax {

class TopksoftmaxInfo {
    TopksoftmaxInfo() = default;

public:
    infiniDtype_t xtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> x_strides;
    size_t N;
    size_t width;

public:
    size_t ndim() const { return shape.size(); }
    size_t dim() const { return shape[ndim() - 1]; }

    static utils::Result<TopksoftmaxInfo> create(infiniopTensorDescriptor_t x_desc) {

        auto xtype = x_desc->dtype();
        if ((xtype != infiniDtype_t::INFINI_DTYPE_F32) && (xtype != infiniDtype_t::INFINI_DTYPE_F16) && (xtype != infiniDtype_t::INFINI_DTYPE_BF16)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (x_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t N = x_desc->shape()[0];     // token数量
        size_t width = x_desc->shape()[1]; // 专家数量

        return utils::Result<TopksoftmaxInfo>(TopksoftmaxInfo{xtype,
                                                              x_desc->shape(),
                                                              x_desc->strides(),
                                                              N,
                                                              width});
    }
};

} // namespace op::topksoftmax

#endif // __TOPKSOFTMAX_INFO_H__
