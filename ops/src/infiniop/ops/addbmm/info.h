#ifndef __ADDBMM_INFO_H__
#define __ADDBMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::addbmm {

class AddbmmInfo {
    AddbmmInfo() = default;

public:
    // 矩阵运算维度的定义:
    // batch1: (b, n, m)
    // batch2: (b, m, p)
    // input/output: (n, p)
    size_t _b;
    size_t _n;
    size_t _m;
    size_t _p;

    // 【新增】步长信息 (Strides)
    // 用于处理非连续内存布局
    std::vector<int64_t> _out_strides; // [stride_n, stride_p]
    std::vector<int64_t> _in_strides;  // [stride_n, stride_p]
    std::vector<int64_t> _b1_strides;  // [stride_b, stride_n, stride_m]
    std::vector<int64_t> _b2_strides;  // [stride_b, stride_m, stride_p]

    // 标量系数
    float _alpha;
    float _beta;

    // 数据类型
    int _dtype;

    // Getters
    size_t b() const { return _b; }
    size_t n() const { return _n; }
    size_t m() const { return _m; }
    size_t p() const { return _p; }

    // 【新增】Strides Getters
    const std::vector<int64_t> &out_strides() const { return _out_strides; }
    const std::vector<int64_t> &in_strides() const { return _in_strides; }
    const std::vector<int64_t> &b1_strides() const { return _b1_strides; }
    const std::vector<int64_t> &b2_strides() const { return _b2_strides; }

    float alpha() const { return _alpha; }
    float beta() const { return _beta; }
    int dtype() const { return _dtype; }

    static utils::Result<AddbmmInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        infiniopTensorDescriptor_t batch1_desc,
        infiniopTensorDescriptor_t batch2_desc,
        float alpha,
        float beta) {

        // 1. 检查数据类型一致性
        int dtype = out_desc->dtype();
        if (in_desc->dtype() != dtype || batch1_desc->dtype() != dtype || batch2_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 2. 检查维度数量 (ndim)
        if (batch1_desc->ndim() != 3 || batch2_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (in_desc->ndim() != 2 || out_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &b1_shape = batch1_desc->shape();
        const auto &b2_shape = batch2_desc->shape();
        const auto &in_shape = in_desc->shape();
        const auto &out_shape = out_desc->shape();

        // 3. 解析并校验维度
        size_t b = b1_shape[0];
        size_t n = b1_shape[1];
        size_t m = b1_shape[2];

        if (b2_shape[0] != b || b2_shape[1] != m) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        size_t p = b2_shape[2];

        if (out_shape[0] != n || out_shape[1] != p) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (in_shape[0] != n || in_shape[1] != p) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 4. 【新增】提取 Strides
        auto out_strides = out_desc->strides();
        auto in_strides = in_desc->strides();
        auto b1_strides = batch1_desc->strides();
        auto b2_strides = batch2_desc->strides();

        // 5. 返回 Info 对象
        AddbmmInfo info;
        info._b = b;
        info._n = n;
        info._m = m;
        info._p = p;
        info._out_strides = out_strides;
        info._in_strides = in_strides;
        info._b1_strides = b1_strides;
        info._b2_strides = b2_strides;
        info._alpha = alpha;
        info._beta = beta;
        info._dtype = dtype;

        return utils::Result<AddbmmInfo>(info);
    }
};

} // namespace op::addbmm

#endif // __ADDBMM_INFO_H__
