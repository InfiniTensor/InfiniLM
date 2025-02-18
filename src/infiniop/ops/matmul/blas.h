#ifndef __BLAS_H__
#define __BLAS_H__

#include "../utils.h"
#include "infiniop/operator.h"
#include <algorithm>
#include <stdint.h>

typedef struct BlasMatrix {
    size_t ndim;
    size_t batch;
    int64_t stride;
    size_t rows;
    size_t cols;
    int64_t row_stride;
    int64_t col_stride;

    BlasMatrix() {}

    BlasMatrix(infiniopTensorDescriptor_t layout, infiniopStatus_t *status) {
        if (layout->ndim == 2) {
            this->ndim = 2;
            this->batch = 1;
            this->stride = 0;
            this->rows = layout->shape[0];
            this->cols = layout->shape[1];
            this->row_stride = layout->strides[0];
            this->col_stride = layout->strides[1];
        } else if (layout->ndim == 3) {
            this->ndim = 3;
            this->batch = layout->shape[0];
            this->stride = this->batch == 1 ? 0 : layout->strides[0];
            this->rows = layout->shape[1];
            this->cols = layout->shape[2];
            this->row_stride = layout->strides[1];
            this->col_stride = layout->strides[2];
        } else {
            *status = INFINIOP_STATUS_BAD_TENSOR_SHAPE;
            return;
        }

        if (this->row_stride != 1 && this->col_stride != 1) {
            *status = INFINIOP_STATUS_BAD_TENSOR_STRIDES;
            return;
        }

        *status = INFINIOP_STATUS_SUCCESS;
    }

    bool match_batch(size_t _batch) const {
        return this->batch == _batch || this->batch == 1;
    }

    void transpose() {
        std::swap(rows, cols);
        std::swap(row_stride, col_stride);
    }

    int64_t ld() const {
        if (this->row_stride == 1) {
            return this->col_stride;
        } else {
            return this->row_stride;
        }
    }
} BlasMatrix;

struct MatmulInfo {
    BlasMatrix a_matrix;
    BlasMatrix b_matrix;
    BlasMatrix c_matrix;

    size_t m, n, k, batch;

    bool is_transed = false;

    MatmulInfo(infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc, infiniopStatus_t *status, bool col_major = true) {
        a_matrix = BlasMatrix(a_desc, status);
        if (*status != INFINIOP_STATUS_SUCCESS) {
            return;
        }
        b_matrix = BlasMatrix(b_desc, status);
        if (*status != INFINIOP_STATUS_SUCCESS) {
            return;
        }
        c_matrix = BlasMatrix(c_desc, status);
        if (*status != INFINIOP_STATUS_SUCCESS) {
            return;
        }

        if (c_matrix.rows != a_matrix.rows || c_matrix.cols != b_matrix.cols || a_matrix.cols != b_matrix.rows) {
            *status = INFINIOP_STATUS_BAD_TENSOR_SHAPE;
            return;
        }

        batch = c_matrix.batch;
        if (!a_matrix.match_batch(batch) || !b_matrix.match_batch(batch)) {
            *status = INFINIOP_STATUS_BAD_TENSOR_SHAPE;
            return;
        }

        if ((col_major && c_matrix.col_stride == 1) || (!col_major && c_matrix.row_stride == 1)) {
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

#endif // __BLAS_H__
