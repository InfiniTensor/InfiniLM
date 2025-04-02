#include "gemm_kunlun.h"
#include "../../../../utils.h"
#include "../../../devices/kunlun/kunlun_handle.h"

namespace op::gemm::kunlun {

typedef device::kunlun::Handle::Internal HandleInternal;

struct Descriptor::Opaque {
    std::shared_ptr<HandleInternal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    auto handle = reinterpret_cast<device::kunlun::Handle *>(handle_);
    auto dtype = c_desc->dtype();

    if (dtype != INFINI_DTYPE_F16 && dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto result = MatmulInfo::create(c_desc, a_desc, b_desc, MatrixLayout::ROW_MAJOR);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype, result.take(), 0,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <class Tdata>
infiniStatus_t calculate(
    MatmulInfo info,
    std::shared_ptr<HandleInternal> internal,
    infiniDtype_t dtype,
    void *c,
    float beta,
    const void *a,
    const void *b,
    float alpha,
    kunlunStream_t stream) {

    if (info.is_transed) {
        std::swap(a, b);
    }

    auto transA = info.a_matrix.col_stride == 1 ? false : true;
    auto transB = info.b_matrix.col_stride == 1 ? false : true;

    auto unit = infiniSizeOf(dtype);

    CHECK_STATUS(internal->useXdnn(
        (kunlunStream_t)stream,
        [&](xdnnHandle_t handle) {
            for (size_t i = 0; i < info.batch; i++) {
                CHECK_XDNN((xdnn::fc_fusion<Tdata, Tdata, Tdata, int16_t>(
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
                    nullptr)));
            }
            return INFINI_STATUS_SUCCESS;
        }));
    return INFINI_STATUS_SUCCESS;
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
        return op::gemm::kunlun::calculate<float16>(_info, _opaque->internal, _dtype, c, beta, a, b, alpha, (kunlunStream_t)stream);
    case INFINI_DTYPE_F32:
        return op::gemm::kunlun::calculate<float>(_info, _opaque->internal, _dtype, c, beta, a, b, alpha, (kunlunStream_t)stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::gemm::kunlun
