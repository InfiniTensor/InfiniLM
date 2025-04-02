#include "gemm_bang.h"
#include "../../../devices/bang/common_bang.h"
#include <cnnl_extra.h>

namespace op::gemm::bang {

struct Descriptor::Opaque {
    cnnlMatMulDescriptor_t op;
    cnnlMatMulAlgo_t algo;
    cnnlMatMulHeuristicResult_t algoResult;
    cnnlTensorDescriptor_t a, b, c;
    std::shared_ptr<device::bang::Handle::Internal> internal;

    ~Opaque() {
        cnnlDestroyTensorDescriptor(a);
        cnnlDestroyTensorDescriptor(b);
        cnnlDestroyTensorDescriptor(c);
        cnnlMatMulDescDestroy(op);
        cnnlMatMulAlgoDestroy(algo);
        cnnlDestroyMatMulHeuristicResult(algoResult);
    }
};

static infiniStatus_t setMatrixTensorEx(
    cnnlTensorDescriptor_t desc,
    const BlasMatrix &matrix, infiniDtype_t dtype,
    bool trans = false) {
    int ndim = matrix.ndim;
    int batch = matrix.batch;
    int stride = static_cast<int>(matrix.stride);
    int rows = matrix.rows;
    int cols = matrix.cols;
    int row_stride = matrix.row_stride;
    int col_stride = matrix.col_stride;

    switch (ndim) {
    case 3: {
        std::vector<int> dim_size = {batch, rows, cols};
        std::vector<int> dim_stride = {stride, row_stride, col_stride};
        CHECK_BANG(cnnlSetTensorDescriptorEx(
            desc, CNNL_LAYOUT_ARRAY,
            device::bang::getCnnlDtype(dtype), dim_size.size(),
            dim_size.data(), dim_stride.data()));
    } break;
    case 2: {
        std::vector<int> dim_size = {rows, cols};
        std::vector<int> dim_stride = {row_stride, col_stride};
        CHECK_BANG(cnnlSetTensorDescriptorEx(
            desc, CNNL_LAYOUT_ARRAY,
            device::bang::getCnnlDtype(dtype), dim_size.size(),
            dim_size.data(), dim_stride.data()));
    } break;
    }
    return INFINI_STATUS_SUCCESS;
}

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::bang::cambricon::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::ROW_MAJOR);
    CHECK_RESULT(result);
    auto info = result.take();

    cnnlTensorDescriptor_t a, b, c;
    CHECK_BANG(cnnlCreateTensorDescriptor(&a));
    CHECK_BANG(cnnlCreateTensorDescriptor(&b));
    CHECK_BANG(cnnlCreateTensorDescriptor(&c));

    CHECK_STATUS(setMatrixTensorEx(a, info.a_matrix, a_desc->dtype()));
    CHECK_STATUS(setMatrixTensorEx(b, info.b_matrix, b_desc->dtype()));
    CHECK_STATUS(setMatrixTensorEx(c, info.c_matrix, c_desc->dtype()));

    cnnlMatMulDescriptor_t op;
    cnnlMatMulAlgo_t algo;
    cnnlMatMulHeuristicResult_t algoResult;
    CHECK_BANG(cnnlMatMulDescCreate(&op));
    CHECK_BANG(cnnlMatMulAlgoCreate(&algo));
    CHECK_BANG(cnnlCreateMatMulHeuristicResult(&algoResult));
    int32_t use_stride = true;
    CHECK_BANG(cnnlSetMatMulDescAttr(
        op,
        CNNL_MATMUL_USE_STRIDE,
        &use_stride,
        sizeof(int32_t)));
    int count = 0;

    CHECK_STATUS(
        handle->internal()->useCnnl(
            (cnrtQueue_t) nullptr,
            [&](cnnlHandle_t _handle) {
                CHECK_BANG(
                    cnnlGetBatchMatMulAlgoHeuristic(
                        _handle,
                        op, a, b, c,
                        NULL, 1, &algoResult, &count));
                return INFINI_STATUS_SUCCESS;
            }));

    size_t workspace_size;
    CHECK_BANG(cnnlGetBatchMatMulHeuristicResult(algoResult, algo, &workspace_size));

    *desc_ptr = new Descriptor(
        dtype, info, workspace_size,
        new Opaque{
            op, algo, algoResult, a, b, c, handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {

    if (_info.is_transed) {
        std::swap(a, b);
    }
    CHECK_STATUS(_opaque->internal->useCnnl(
        (cnrtQueue_t)stream,
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlBatchMatMulBCast_v2(
                handle,
                _opaque->op,
                _opaque->algo,
                &alpha,
                _opaque->a, a,
                _opaque->b, b,
                &beta,
                _opaque->c, c,
                workspace,
                workspace_size));
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync((cnrtQueue_t)stream);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gemm::bang
