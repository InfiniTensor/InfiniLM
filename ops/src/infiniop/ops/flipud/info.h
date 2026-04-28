#ifndef __FLIPUD_INFO_H__
#define __FLIPUD_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::flipud {

class FlipudInfo {
    FlipudInfo() = default;

public:
    int _dtype;
    int _ndim;
    size_t _numel;

    int dtype() const { return _dtype; }
    int ndim() const { return _ndim; }
    size_t numel() const { return _numel; }

    FlipudInfo(int dtype, int ndim, size_t numel)
        : _dtype(dtype), _ndim(ndim), _numel(numel) {}

    static utils::Result<FlipudInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc) {

        if (out_desc->dtype() != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (out_desc->ndim() != input_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (input_desc->ndim() < 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &in_shape = input_desc->shape();
        const auto &out_shape = out_desc->shape();

        for (size_t i = 0; i < input_desc->ndim(); ++i) {
            if (in_shape[i] != out_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        return utils::Result<FlipudInfo>(FlipudInfo{
            input_desc->dtype(),
            static_cast<int>(input_desc->ndim()),
            input_desc->numel()});
    }
};

} // namespace op::flipud

#endif // __FLIPUD_INFO_H__
