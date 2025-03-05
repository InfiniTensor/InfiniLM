#include "matmul_kunlun.h"
#include "../../../devices/kunlun/common_kunlun.h"
#include "../../utils.h"

namespace matmul::kunlun {

struct Descriptor::Opaque {
    std::shared_ptr<Pool<xdnnHandle_t>> xdnn_handle_pool;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t c_desc,
                                  infiniopTensorDescriptor_t a_desc,
                                  infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<infiniopKunlunHandle_t>(handle_);
    auto dtype = c_desc->dtype;

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    infiniStatus_t status;
    auto info = MatmulInfo(c_desc, a_desc, b_desc, &status, MatrixLayout::ROW_MAJOR);
    if (status != INFINI_STATUS_SUCCESS) {
        return status;
    }

    *desc_ptr = new Descriptor(
        dtype, info, 0,
        new Opaque{handle->xdnn_handle_pool},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <class Tdata>
void calculate(
    const MatmulInfo &info,
    std::shared_ptr<Pool<xdnnHandle_t>> &xdnn_handle_pool,
    infiniDtype_t dtype,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    KunlunStream_t stream) {

    if (info.is_transed) {
        std::swap(a, b);
    }

    auto transA = info.a_matrix.col_stride == 1 ? false : true;
    auto transB = info.b_matrix.col_stride == 1 ? false : true;

    auto unit = infiniSizeOf(dtype);

    use_xdnn(xdnn_handle_pool,
             (KunlunStream_t)stream,
             [&](xdnnHandle_t handle) {
                 for (size_t i = 0; i < info.batch; i++) {
                     xdnn::fc_fusion<Tdata, Tdata, Tdata, int16_t>(
                         handle,
                         (Tdata *)((char *)a + i * info.a_matrix.stride * unit),
                         (Tdata *)((char *)b + i * info.b_matrix.stride * unit),
                         (Tdata *)((char *)c + i * info.c_matrix.stride * unit),
                         info.m,
                         info.n,
                         info.k,
                         transA,
                         transB,
                         nullptr,
                         nullptr,
                         nullptr,
                         info.a_matrix.ld(),
                         info.b_matrix.ld(),
                         info.c_matrix.ld(),
                         alpha,
                         beta,
                         nullptr,
                         xdnn::Activation_t::LINEAR,
                         nullptr);
                 }
             });
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t worksapce_size,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    void *stream) const {
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        kunlun::calculate<float16>(_info, _opaque->xdnn_handle_pool, _dtype, c, beta, a, b, alpha, (KunlunStream_t)stream);
        return INFINI_STATUS_SUCCESS;

    case INFINI_DTYPE_F32:
        kunlun::calculate<float>(_info, _opaque->xdnn_handle_pool, _dtype, c, beta, a, b, alpha, (KunlunStream_t)stream);
        return INFINI_STATUS_SUCCESS;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace matmul::kunlun
