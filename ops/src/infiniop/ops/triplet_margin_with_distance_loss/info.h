#ifndef __TRIPLET_MARGIN_WITH_DISTANCE_LOSS_INFO_H__
#define __TRIPLET_MARGIN_WITH_DISTANCE_LOSS_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::triplet_margin_with_distance_loss {

class TripletMarginWithDistanceLossInfo {
    TripletMarginWithDistanceLossInfo() = default;

public:
    int _dtype;
    float _margin;
    int _swap;
    int _reduction;
    size_t _num_elements;

    int dtype() const { return _dtype; }
    float margin() const { return _margin; }
    int swap() const { return _swap; }
    int reduction() const { return _reduction; }
    size_t num_elements() const { return _num_elements; }

    TripletMarginWithDistanceLossInfo(int dtype, float margin, int swap, int reduction, size_t num_elements)
        : _dtype(dtype), _margin(margin), _swap(swap), _reduction(reduction), _num_elements(num_elements) {}

    static utils::Result<TripletMarginWithDistanceLossInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t anchor_desc,
        infiniopTensorDescriptor_t positive_desc,
        infiniopTensorDescriptor_t negative_desc,
        float margin,
        int swap,
        int reduction) {

        // 1. Validate Dtypes
        int dtype = anchor_desc->dtype();
        if (positive_desc->dtype() != dtype || negative_desc->dtype() != dtype || output_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 2. Validate Input Shapes
        // FIX: 使用 size_t 接收 ndim 以避免符号比较警告
        size_t ndim = anchor_desc->ndim();
        if (positive_desc->ndim() != ndim || negative_desc->ndim() != ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t num_elements = 1;
        // FIX: 循环变量使用 size_t
        for (size_t i = 0; i < ndim; ++i) {
            auto dim_size = anchor_desc->shape()[i];
            if (positive_desc->shape()[i] != dim_size || negative_desc->shape()[i] != dim_size) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            num_elements *= dim_size;
        }

        // 3. Validate Output Shape based on Reduction
        if (reduction == 0) { // None
            if (output_desc->ndim() != ndim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            for (size_t i = 0; i < ndim; ++i) {
                if (output_desc->shape()[i] != anchor_desc->shape()[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        } else { // Mean or Sum
            size_t output_size = 1;
            // FIX: output_desc->ndim() 返回 size_t，循环变量 i 也应为 size_t
            for (size_t i = 0; i < output_desc->ndim(); ++i) {
                output_size *= output_desc->shape()[i];
            }
            if (output_size != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        return utils::Result<TripletMarginWithDistanceLossInfo>(TripletMarginWithDistanceLossInfo{
            dtype,
            margin,
            swap,
            reduction,
            num_elements});
    }
};

} // namespace op::triplet_margin_with_distance_loss

#endif // __TRIPLET_MARGIN_WITH_DISTANCE_LOSS_INFO_H__
