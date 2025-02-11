#ifndef __CNNL_MATMUL_H__
#define __CNNL_MATMUL_H__
#include "../../../devices/bang/bang_handle.h"
#include "../blas.h"
#include "cnnl.h"
#include "cnnl_extra.h"
#include "operators.h"

struct MatmulBangDescriptor {
    Device device;
    int device_id;
    MatmulInfo info;
    float alpha;
    float beta;
    DT dtype;
    std::shared_ptr<Pool<cnnlHandle_t>> cnnl_handles;
    cnnlTensorDescriptor_t aDesc;
    cnnlTensorDescriptor_t bDesc;
    cnnlTensorDescriptor_t cDesc;
    cnnlMatMulDescriptor_t opDesc;
    cnnlMatMulAlgo_t algo;
    cnnlMatMulHeuristicResult_t algoResult;
};
typedef struct MatmulBangDescriptor *MatmulBangDescriptor_t;

infiniopStatus_t bangCreateMatmulDescriptor(BangHandle_t handle,
                                            MatmulBangDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            float alpha,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc,
                                            float beta);

infiniopStatus_t bangGetMatmulWorkspaceSize(MatmulBangDescriptor_t desc, uint64_t *size);

infiniopStatus_t bangMatmul(MatmulBangDescriptor_t desc, void *workspace, uint64_t workspace_size, void *c, void const *a, void const *b, void *stream);

infiniopStatus_t bangDestroyMatmulDescriptor(MatmulBangDescriptor_t desc);

inline void setMatrixTensorEx(cnnlTensorDescriptor_t desc, const BlasMatrix &matrix, bool trans = false) {
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
        cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                                  dim_size.size(), dim_size.data(), dim_stride.data());
    } else if (ndim == 2) {
        std::vector<int> dim_size = {rows, cols};
        std::vector<int> dim_stride = {row_stride, col_stride};
        cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_HALF,
                                  dim_size.size(), dim_size.data(), dim_stride.data());
    }
}


#endif// __CNNL_MATMUL_H__
