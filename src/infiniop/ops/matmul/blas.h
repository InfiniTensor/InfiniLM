#ifndef __BLAS_H__
#define __BLAS_H__

#include "../../operator.h"
#include "../../tensor.h"
#include <algorithm>

namespace op::matmul {

struct BlasMatrix {
    size_t ndim;
    size_t batch;
    ptrdiff_t stride;
    size_t rows;
    size_t cols;
    ptrdiff_t row_stride;
    ptrdiff_t col_stride;

    BlasMatrix() = default;

    BlasMatrix(infiniopTensorDescriptor_t layout, infiniStatus_t *status) {
        if (layout->ndim() == 2) {
            ndim = 2;
            batch = 1;
            stride = 0;
            rows = layout->dim(0);
            cols = layout->dim(1);
            row_stride = layout->stride(0);
            col_stride = layout->stride(1);
        } else if (layout->ndim() == 3) {
            ndim = 3;
            batch = layout->dim(0);
            stride = batch == 1 ? 0 : layout->stride(0);
            rows = layout->dim(1);
            cols = layout->dim(2);
            row_stride = layout->stride(1);
            col_stride = layout->stride(2);
        } else {
            *status = INFINI_STATUS_BAD_TENSOR_SHAPE;
            return;
        }

        if (row_stride != 1 && col_stride != 1) {
            *status = INFINI_STATUS_BAD_TENSOR_STRIDES;
            return;
        }

        *status = INFINI_STATUS_SUCCESS;
    }

    bool match_batch(size_t _batch) const {
        return batch == _batch || batch == 1;
    }

    void transpose() {
        std::swap(rows, cols);
        std::swap(row_stride, col_stride);
    }

    ptrdiff_t ld() const {
        return row_stride == 1 ? col_stride : row_stride;
    }
};

enum class MatrixLayout : char {
    COL_MAJOR,
    ROW_MAJOR,
};

struct MatmulInfo {
    BlasMatrix a_matrix;
    BlasMatrix b_matrix;
    BlasMatrix c_matrix;

    size_t m, n, k, batch;

    bool is_transed = false;

    MatmulInfo(infiniopTensorDescriptor_t c_desc,
               infiniopTensorDescriptor_t a_desc,
               infiniopTensorDescriptor_t b_desc,
               infiniStatus_t *status,
               MatrixLayout layout) {
        a_matrix = BlasMatrix(a_desc, status);
        if (*status != INFINI_STATUS_SUCCESS) {
            return;
        }
        b_matrix = BlasMatrix(b_desc, status);
        if (*status != INFINI_STATUS_SUCCESS) {
            return;
        }
        c_matrix = BlasMatrix(c_desc, status);
        if (*status != INFINI_STATUS_SUCCESS) {
            return;
        }

        if (c_matrix.rows != a_matrix.rows || c_matrix.cols != b_matrix.cols || a_matrix.cols != b_matrix.rows) {
            *status = INFINI_STATUS_BAD_TENSOR_SHAPE;
            return;
        }

        batch = c_matrix.batch;
        if (!a_matrix.match_batch(batch) || !b_matrix.match_batch(batch)) {
            *status = INFINI_STATUS_BAD_TENSOR_SHAPE;
            return;
        }

        if ((layout == MatrixLayout::COL_MAJOR && c_matrix.col_stride == 1)
            || (layout == MatrixLayout::ROW_MAJOR && c_matrix.row_stride == 1)) {
            c_matrix.transpose();
            b_matrix.transpose();
            a_matrix.transpose();
            std::swap(a_matrix, b_matrix);
            is_transed = true;
        }

        m = c_matrix.rows;
        n = c_matrix.cols;
        k = a_matrix.cols;
    }
};

} // namespace op::matmul

#endif // __BLAS_H__
