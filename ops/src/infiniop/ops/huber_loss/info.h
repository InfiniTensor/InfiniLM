#ifndef __HUBER_LOSS_INFO_H__
#define __HUBER_LOSS_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::huber_loss {

class HuberLossInfo {
    HuberLossInfo() = default;

public:
    int _dtype;
    float _delta;
    int _reduction;
    size_t _count;

    int dtype() const { return _dtype; }
    float delta() const { return _delta; }
    int reduction() const { return _reduction; }
    size_t count() const { return _count; }

    HuberLossInfo(int dtype, float delta, int reduction, size_t count)
        : _dtype(dtype), _delta(delta), _reduction(reduction), _count(count) {}

    static utils::Result<HuberLossInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t target_desc,
        float delta,
        int reduction) {

        if (input_desc->ndim() != target_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t total_count = input_desc->numel();
        if (target_desc->numel() != total_count) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        for (size_t i = 0; i < input_desc->ndim(); ++i) {
            if (input_desc->shape()[i] != target_desc->shape()[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        if (input_desc->dtype() != target_desc->dtype() || input_desc->dtype() != out_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (reduction == 0) {
            if (out_desc->ndim() != input_desc->ndim()) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            for (size_t i = 0; i < out_desc->ndim(); ++i) {
                if (out_desc->shape()[i] != input_desc->shape()[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        } else {
            if (out_desc->numel() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        return utils::Result<HuberLossInfo>(HuberLossInfo{
            input_desc->dtype(),
            delta,
            reduction,
            total_count});
    }
};

} // namespace op::huber_loss

#endif // __HUBER_LOSS_INFO_H__
