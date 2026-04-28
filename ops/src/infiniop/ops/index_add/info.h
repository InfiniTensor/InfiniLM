#ifndef __INDEX_ADD_INFO_H__
#define __INDEX_ADD_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::index_add {

class IndexAddInfo {
    IndexAddInfo() = default;

public:
    int _dtype;     // 数据类型 (Input/Output/Source)
    int _idx_dtype; // 索引类型 (int32, int64)
    int64_t _dim;   // 操作维度
    float _alpha;   // 缩放因子

    // 【新增】几何信息，用于计算内存偏移
    size_t _outer_size; // dim 左侧维度的乘积
    size_t _inner_size; // dim 右侧维度的乘积
    size_t _dim_size;   // Input/Output 在 dim 维度的长度
    size_t _index_len;  // Index 的长度

    // 【修改】构造函数，初始化新增成员
    IndexAddInfo(int dtype, int idx_dtype, int64_t dim, float alpha,
                 size_t outer_size, size_t inner_size, size_t dim_size, size_t index_len)
        : _dtype(dtype), _idx_dtype(idx_dtype), _dim(dim), _alpha(alpha),
          _outer_size(outer_size), _inner_size(inner_size), _dim_size(dim_size), _index_len(index_len) {}

    int dtype() const { return _dtype; }
    int idx_dtype() const { return _idx_dtype; }
    int64_t dim() const { return _dim; }
    float alpha() const { return _alpha; }

    // 【新增】Getter 方法
    size_t outer_size() const { return _outer_size; }
    size_t inner_size() const { return _inner_size; }
    size_t dim_size() const { return _dim_size; }
    size_t index_len() const { return _index_len; }

    static utils::Result<IndexAddInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        int64_t dim,
        infiniopTensorDescriptor_t index_desc,
        infiniopTensorDescriptor_t source_desc,
        float alpha) {

        // 1. 检查数据类型一致性 (Output vs Input vs Source)
        int dtype = in_desc->dtype();
        if (out_desc->dtype() != dtype || source_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 2. 检查索引数据类型
        int idx_dtype = index_desc->dtype();
        if (idx_dtype != INFINI_DTYPE_I32 && idx_dtype != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 3. 检查维度有效性
        int64_t ndim = static_cast<int64_t>(in_desc->ndim());
        if (dim < 0 || dim >= ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 4. 检查 Index 形状
        if (index_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 【新增】计算几何信息
        const auto &in_shape = in_desc->shape();

        // outer_size: dim 之前所有维度的乘积
        size_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= in_shape[i];
        }

        // inner_size: dim 之后所有维度的乘积 (即 stride)
        size_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) {
            inner_size *= in_shape[i];
        }

        // dim_size
        size_t dim_size = in_shape[dim];

        // index_len
        size_t index_len = index_desc->shape()[0];

        // 5. 检查 Source 形状一致性
        // 规则: [Outer, IndexLen, Inner]
        if (source_desc->ndim() != in_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &src_shape = source_desc->shape();

        for (int64_t i = 0; i < ndim; ++i) {
            if (i == dim) {
                if (src_shape[i] != index_len) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            } else {
                if (src_shape[i] != in_shape[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        }

        // 6. 检查 Output 与 Input 形状一致性
        if (out_desc->ndim() != in_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const auto &out_shape = out_desc->shape();
        for (int64_t i = 0; i < ndim; ++i) {
            if (out_shape[i] != in_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 7. 返回 Info 对象 (包含计算好的几何信息)
        return utils::Result<IndexAddInfo>(IndexAddInfo{
            dtype,
            idx_dtype,
            dim,
            alpha,
            outer_size, // pass
            inner_size, // pass
            dim_size,   // pass
            index_len   // pass
        });
    }
};

} // namespace op::index_add

#endif // __INDEX_ADD_INFO_H__
