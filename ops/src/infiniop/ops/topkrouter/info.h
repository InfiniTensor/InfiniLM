#ifndef __TOPKROUTER_INFO_H__
#define __TOPKROUTER_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::topkrouter {

class TopkrouterInfo {
    TopkrouterInfo() = default;

public:
    infiniDtype_t xtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> x_strides;
    size_t N;
    size_t width;

public:
    size_t ndim() const { return shape.size(); }
    size_t dim() const { return shape[ndim() - 1]; }

    static utils::Result<TopkrouterInfo> create(infiniopTensorDescriptor_t x_desc) {

        auto xtype = x_desc->dtype();
        if ((xtype != infiniDtype_t::INFINI_DTYPE_F32) && (xtype != infiniDtype_t::INFINI_DTYPE_F16) && (xtype != infiniDtype_t::INFINI_DTYPE_BF16)) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        size_t N = x_desc->shape()[0];     // token数量
        size_t width = x_desc->shape()[1]; // 专家数量

        if (x_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<TopkrouterInfo>(TopkrouterInfo{xtype, x_desc->shape(), x_desc->strides(), N, width});
    }
};

} // namespace op::topkrouter

#endif // __Topkrouter_INFO_H__
