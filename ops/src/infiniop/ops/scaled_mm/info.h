#ifndef __I8GEMM_INFO_H__
#define __I8GEMM_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <algorithm>

namespace op::i8gemm {

struct BlasMatrix {
    int ndim;
    int batch;
    int stride;

    int rows;
    int cols;
    int row_stride;
    int col_stride;

    static utils::Result<BlasMatrix> create(infiniopTensorDescriptor_t layout) {
        BlasMatrix ans;

        if (layout->ndim() == 2) {
            ans.ndim = 2;
            ans.batch = 1;
            ans.stride = 0;
            ans.rows = layout->dim(0);
            ans.cols = layout->dim(1);
            ans.row_stride = layout->stride(0);
            ans.col_stride = layout->stride(1);
        } else if (layout->ndim() == 3) {
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

        if (ans.row_stride != 1 && ans.col_stride != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<BlasMatrix>(ans);
    }

    bool match_batch(int _batch) const {
        return batch == _batch || batch == 1;
    }

    void transpose() {
        std::swap(rows, cols);
        std::swap(row_stride, col_stride);
    }

    int ld() const {
        return row_stride == 1 ? col_stride : row_stride;
    }
};

enum class MatrixLayout : char {
    COL_MAJOR,
    ROW_MAJOR,
};

class I8GemmInfo {
    I8GemmInfo() = default;

public:
    BlasMatrix a_matrix;
    BlasMatrix b_matrix;
    BlasMatrix out_matrix;

    int m, n, k, batch;

    static utils::Result<I8GemmInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        MatrixLayout layout) {

        auto a_matrix = BlasMatrix::create(a_desc);
        CHECK_RESULT(a_matrix);

        auto b_matrix = BlasMatrix::create(b_desc);
        CHECK_RESULT(b_matrix);

        auto out_matrix = BlasMatrix::create(out_desc);
        CHECK_RESULT(out_matrix);

        if (out_matrix->rows != a_matrix->rows || out_matrix->cols != b_matrix->cols || a_matrix->cols != b_matrix->rows) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto batch = out_matrix->batch;
        if (!a_matrix->match_batch(batch) || !b_matrix->match_batch(batch)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto m = out_matrix->rows;
        auto n = out_matrix->cols;
        auto k = a_matrix->cols;

        return utils::Result<I8GemmInfo>(I8GemmInfo{
            a_matrix.take(),
            b_matrix.take(),
            out_matrix.take(),
            m,
            n,
            k,
            batch});
    }
};

} // namespace op::i8gemm

#endif // __I8GEMM_INFO_H__
