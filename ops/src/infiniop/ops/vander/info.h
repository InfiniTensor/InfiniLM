#ifndef __VANDER_INFO_H__
#define __VANDER_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::vander {

class VanderInfo {
    VanderInfo() = default;

public:
    int _dtype;       // 数据类型
    bool _increasing; // 幂次顺序 (false: 递减, true: 递增)

    // 形状信息缓存
    size_t _rows; // 输入向量长度 (N)
    size_t _cols; // 输出矩阵列数 (M)

    int dtype() const { return _dtype; }
    bool increasing() const { return _increasing; }
    size_t rows() const { return _rows; }
    size_t cols() const { return _cols; }

    // 构造函数
    VanderInfo(int dtype, bool increasing, size_t rows, size_t cols)
        : _dtype(dtype), _increasing(increasing), _rows(rows), _cols(cols) {}

    static utils::Result<VanderInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        int N,            // 用户指定的列数，若 <= 0 则默认为输入长度
        int increasing) { // C API 传入的是 int，内部转为 bool

        // 1. 检查输入形状
        // Input 必须是 1D 向量: (rows)
        if (input_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t rows = input_desc->shape()[0];
        size_t cols = (N > 0) ? static_cast<size_t>(N) : rows;

        // 3. 检查输出形状
        // Output 必须是 2D 矩阵: (rows, cols)
        if (out_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (out_desc->shape()[0] != rows) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (out_desc->shape()[1] != cols) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (out_desc->dtype() != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        return utils::Result<VanderInfo>(VanderInfo{
            input_desc->dtype(),           // _dtype
            static_cast<bool>(increasing), // _increasing
            rows,                          // _rows
            cols                           // _cols
        });
    }
};

} // namespace op::vander

#endif // __VANDER_INFO_H__
