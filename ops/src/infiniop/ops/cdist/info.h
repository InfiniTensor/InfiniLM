#ifndef __CDIST_INFO_H__
#define __CDIST_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <algorithm>

namespace op::cdist {

/**
 * 借用 BlasMatrix 的概念来描述 cdist 的输入输出矩阵
 * x1: (Batch, M, D)
 * x2: (Batch, N, D)
 * y:  (Batch, M, N)
 */
struct CdistMatrix {
    size_t ndim;
    size_t batch;
    ptrdiff_t stride; // Batch 之间的步长
    size_t rows;      // M 或 N
    size_t cols;      // D (特征维度) 或结果中的 N
    ptrdiff_t row_stride;
    ptrdiff_t col_stride;

    static utils::Result<CdistMatrix> create(infiniopTensorDescriptor_t layout) {
        CdistMatrix ans;
        auto ndim = layout->ndim();

        if (ndim == 2) {
            ans.ndim = 2;
            ans.batch = 1;
            ans.stride = 0;
            ans.rows = layout->dim(0);
            ans.cols = layout->dim(1);
            ans.row_stride = layout->stride(0);
            ans.col_stride = layout->stride(1);
        } else if (ndim == 3) {
            ans.ndim = 3;
            ans.batch = layout->dim(0);
            ans.stride = ans.batch == 1 ? 0 : layout->stride(0);
            ans.rows = layout->dim(1);
            ans.cols = layout->dim(2);
            ans.row_stride = layout->stride(1);
            ans.col_stride = layout->stride(2);
        } else {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<CdistMatrix>(ans);
    }

    bool match_batch(size_t _batch) const {
        return batch == _batch || batch == 1;
    }
};

class CdistInfo {
    CdistInfo() = default;

public:
    CdistMatrix x1_matrix;
    CdistMatrix x2_matrix;
    CdistMatrix y_matrix;

    size_t m, n, d, batch;

    static utils::Result<CdistInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x1_desc,
        infiniopTensorDescriptor_t x2_desc) {

        auto x1_res = CdistMatrix::create(x1_desc);
        CHECK_RESULT(x1_res);

        auto x2_res = CdistMatrix::create(x2_desc);
        CHECK_RESULT(x2_res);

        auto y_res = CdistMatrix::create(y_desc);
        CHECK_RESULT(y_res);

        auto x1 = x1_res.take();
        auto x2 = x2_res.take();
        auto y = y_res.take();

        // 1. 维度校验
        // x1(M, D), x2(N, D) -> y(M, N)
        if (x1.cols != x2.cols) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE; // 特征维度 D 必须一致
        }
        if (y.rows != x1.rows || y.cols != x2.rows) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE; // 输出形状必须为 M x N
        }

        // 2. Batch 校验
        size_t batch_size = y.batch;
        if (!x1.match_batch(batch_size) || !x2.match_batch(batch_size)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t m = x1.rows;
        size_t n = x2.rows;
        size_t d = x1.cols;

        return utils::Result<CdistInfo>(CdistInfo{
            x1, x2, y,
            m, n, d, batch_size});
    }
};

} // namespace op::cdist

#endif // __CDIST_INFO_H__
