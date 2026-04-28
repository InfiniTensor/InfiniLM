#include "../../../devices/moore/moore_handle.h"
#include "../../../handle.h"
#include "softplus_moore.h"
#include "softplus_moore_kernel.h"
#include <musa_bf16.h>
#include <musa_fp16.h>

namespace op::softplus::moore {

static constexpr int MAX_DIMS = 8;

struct TensorMetadata {
    int ndim;
    int64_t shape[MAX_DIMS];
    int64_t strides[MAX_DIMS];
};

// ==================================================================
// Kernel 1: 连续内存路径
// ==================================================================
template <typename T>
__global__ void softplus_kernel_contiguous(
    T *output,
    const T *input,
    size_t n,
    float beta,
    float threshold) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        op::softplus::moore::SoftplusOp functor;
        output[idx] = functor(input[idx], beta, threshold);
    }
}

// ==================================================================
// Kernel 2: 非连续内存路径 (Strided)
// ==================================================================
template <typename T>
__global__ void softplus_kernel_strided(
    T *output,
    const T *input,
    size_t n,
    float beta,
    float threshold,
    TensorMetadata meta) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        size_t input_offset = 0;
        size_t temp_idx = idx;

#pragma unroll
        for (int d = meta.ndim - 1; d >= 0; --d) {
            size_t dim_size = meta.shape[d];
            size_t coord = temp_idx % dim_size;
            temp_idx /= dim_size;
            input_offset += coord * meta.strides[d];
        }

        op::softplus::moore::SoftplusOp functor;
        output[idx] = functor(input[input_offset], beta, threshold);
    }
}

// ==================================================================
// Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const SoftplusInfo &info,
    void *stream) {

    size_t n = info.num_elements();
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    if (info.is_contiguous()) {
        softplus_kernel_contiguous<T><<<grid, block, 0, musa_stream>>>(
            reinterpret_cast<T *>(output),
            reinterpret_cast<const T *>(input),
            n,
            info.beta(),
            info.threshold());
    } else {
        TensorMetadata meta;
        meta.ndim = info.ndim();

        const auto &shape_vec = info.shape();
        const auto &stride_vec = info.strides();

        for (int i = 0; i < meta.ndim; ++i) {
            meta.shape[i] = shape_vec[i];
            meta.strides[i] = stride_vec[i];
        }

        softplus_kernel_strided<T><<<grid, block, 0, musa_stream>>>(
            reinterpret_cast<T *>(output),
            reinterpret_cast<const T *>(input),
            n,
            info.beta(),
            info.threshold(),
            meta);
    }
}

// ==================================================================
// Descriptor Implementation
// ==================================================================
struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    float beta,
    float threshold) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    auto result = SoftplusInfo::create(out_desc, input_desc, beta, threshold);
    if (!result) {
        return result.status();
    }

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    switch (_info.dtype()) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<__mt_bfloat16>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::softplus::moore
