#ifndef __CNNL_MATMUL_H__
#define __CNNL_MATMUL_H__
#include "../../../devices/bang/common_bang.h"
#include "../blas.h"
#include "cnnl_extra.h"

struct InfiniopMatmulBangDescriptor {
    infiniDevice_t device;
    int device_id;
    MatmulInfo info;
    infiniDtype_t dtype;
    std::shared_ptr<Pool<cnnlHandle_t>> cnnl_handle_pool;
    cnnlTensorDescriptor_t aDesc;
    cnnlTensorDescriptor_t bDesc;
    cnnlTensorDescriptor_t cDesc;
    cnnlMatMulDescriptor_t opDesc;
    cnnlMatMulAlgo_t algo;
    cnnlMatMulHeuristicResult_t algoResult;
    size_t workspace_size;
};

inline void setMatrixTensorEx(cnnlTensorDescriptor_t desc,
                              const BlasMatrix &matrix, infiniDtype_t dtype,
                              bool trans = false) {
    int ndim = matrix.ndim;
    int batch = matrix.batch;
    int stride = static_cast<int>(matrix.stride);
    int rows = matrix.rows;
    int cols = matrix.cols;
    int row_stride = matrix.row_stride;
    int col_stride = matrix.col_stride;

    if (ndim == 3) {
        std::vector<int> dim_size = {batch, rows, cols};
        std::vector<int> dim_stride = {stride, row_stride, col_stride};
        cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY,
                                  cnnlDataTypeConvert(dtype), dim_size.size(),
                                  dim_size.data(), dim_stride.data());
    } else if (ndim == 2) {
        std::vector<int> dim_size = {rows, cols};
        std::vector<int> dim_stride = {row_stride, col_stride};
        cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY,
                                  cnnlDataTypeConvert(dtype), dim_size.size(),
                                  dim_size.data(), dim_stride.data());
    }
}

#endif // __CNNL_MATMUL_H__
