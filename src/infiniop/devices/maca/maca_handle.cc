#include "common_maca.h"

namespace device::maca {
Handle::Handle(infiniDevice_t device, int device_id)
    : InfiniopHandle{device, device_id},
      _internal(std::make_shared<Handle::Internal>()) {}

auto Handle::internal() const -> const std::shared_ptr<Internal> & {
    return _internal;
}

infiniStatus_t Handle::Internal::use_mcblas(hcStream_t stream, const Fn<hcblasHandle_t> &f) const {
    auto handle = mcblas_handles.pop();
    if (!handle) {
        CHECK_MCBLAS(hcblasCreate(&(*handle)));
    }
    CHECK_MCBLAS(hcblasSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    mcblas_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Handle::Internal::use_mcdnn(hcStream_t stream, const Fn<hcdnnHandle_t> &f) const {
    auto handle = mcdnn_handles.pop();
    if (!handle) {
        CHECK_MCDNN(hcdnnCreate(&(*handle)));
    }
    CHECK_MCDNN(hcdnnSetStream(*handle, stream));
    CHECK_STATUS(f(*handle));
    mcdnn_handles.push(std::move(*handle));
    return INFINI_STATUS_SUCCESS;
}

hcdnnDataType_t getHcdnnDtype(infiniDtype_t dt) {
    switch (dt) {
    case INFINI_DTYPE_F16:
        return HCDNN_DATA_HALF;
    case INFINI_DTYPE_F32:
        return HCDNN_DATA_FLOAT;
    case INFINI_DTYPE_F64:
        return HCDNN_DATA_DOUBLE;
    case INFINI_DTYPE_BF16:
        return HCDNN_DATA_BFLOAT16;
    case INFINI_DTYPE_I8:
        return HCDNN_DATA_INT8;
    case INFINI_DTYPE_I32:
        return HCDNN_DATA_INT32;
    case INFINI_DTYPE_I64:
        return HCDNN_DATA_INT64;
    case INFINI_DTYPE_U8:
        return HCDNN_DATA_UINT8;
    default:
        return HCDNN_DATA_FLOAT;
    }
}

infiniStatus_t Handle::create(InfiniopHandle **handle_ptr, int device_id) {
    *handle_ptr = new Handle(INFINI_DEVICE_METAX, device_id);
    return INFINI_STATUS_SUCCESS;
}

} // namespace device::maca
