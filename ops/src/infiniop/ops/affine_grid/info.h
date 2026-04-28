#ifndef __AFFINE_GRID_INFO_H__
#define __AFFINE_GRID_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::affine_grid {

class AffineGridInfo {
    AffineGridInfo() = default;

public:
    size_t _batch;
    size_t _height;
    size_t _width;
    bool _align_corners;
    int _dtype;

    size_t batch() const { return _batch; }
    size_t height() const { return _height; }
    size_t width() const { return _width; }
    bool align_corners() const { return _align_corners; }
    int dtype() const { return _dtype; }

    static utils::Result<AffineGridInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        bool align_corners) {

        // 1. 检查数据类型一致性
        if (out_desc->dtype() != in_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 2. 检查输入 Theta 的形状
        // 标准 2D Affine Grid 输入必须是 (N, 2, 3)
        if (in_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (in_desc->shape()[1] != 2 || in_desc->shape()[2] != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 3. 检查输出 Grid 的形状
        // 标准 2D Affine Grid 输出必须是 (N, H, W, 2)
        if (out_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        // 最后一维必须是 2 (代表 x, y 坐标)
        if (out_desc->shape()[3] != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 4. 检查 Batch Size 是否匹配
        if (in_desc->shape()[0] != out_desc->shape()[0]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 5. 提取维度信息
        size_t batch = out_desc->shape()[0];
        size_t height = out_desc->shape()[1];
        size_t width = out_desc->shape()[2];
        int dtype = in_desc->dtype();

        // 6. 返回 Info 对象
        return utils::Result<AffineGridInfo>(AffineGridInfo{
            batch,
            height,
            width,
            align_corners,
            dtype});
    }
};

} // namespace op::affine_grid

#endif // __AFFINE_GRID_INFO_H__
