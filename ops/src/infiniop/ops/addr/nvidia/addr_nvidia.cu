
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cu"
#include "addr_nvidia.cuh"
#include "infinicore.h"
#include <cmath>

template <typename Tdata>
INFINIOP_CUDA_KERNEL Addr(Tdata *output, const Tdata *input, const Tdata *vec1,
                          const Tdata *vec2, size_t n, size_t m, float beta,
                          float alpha, ptrdiff_t stride1, ptrdiff_t stride2,
                          ptrdiff_t out_stride_0, ptrdiff_t out_stride_1,
                          ptrdiff_t in_stride_0, ptrdiff_t in_stride_1) {
    addr_kernel<Tdata>(output, input, vec1, vec2, n, m, beta, alpha, stride1,
                       stride2, out_stride_0, out_stride_1, in_stride_0,
                       in_stride_1);
}
namespace op::addr::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(infiniopHandle_t handle,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t out_desc,
                                  infiniopTensorDescriptor_t input_desc,
                                  infiniopTensorDescriptor_t vec1_desc,
                                  infiniopTensorDescriptor_t vec2_desc,
                                  float beta, float alpha) {

    auto info = AddrInfo::create(input_desc, out_desc, vec1_desc, vec2_desc, beta, alpha);
    CHECK_RESULT(info);
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        info.take(), workspace_size,
        new Opaque{
            reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(infiniDtype_t dtype, void *output,
                            const void *input, const void *vec1,
                            const void *vec2, size_t n, size_t m, float beta,
                            float alpha, ptrdiff_t stride1, ptrdiff_t stride2,
                            ptrdiff_t out_stride_0, ptrdiff_t out_stride_1,
                            ptrdiff_t in_stride_0, ptrdiff_t in_stride_1,
                            cudaStream_t stream) {
    unsigned int dn = std::min((unsigned int)n, BLOCK_SIZE),
                 dm = std::min((unsigned int)m, BLOCK_SIZE);
    dim3 grid = {(unsigned int)((n + dn - 1) / dn),
                 (unsigned int)((m + dm - 1) / dm)};
    dim3 block = {dn, dm};
    switch (dtype) {
    case INFINI_DTYPE_F32:
        Addr<float><<<grid, block, 0, stream>>>(
            (float *)output, (const float *)input, (const float *)vec1,
            (const float *)vec2, n, m, beta, alpha, stride1, stride2, out_stride_0,
            out_stride_1, in_stride_0, in_stride_1);
        break;
    case INFINI_DTYPE_F16:
        Addr<__half><<<grid, block, 0, stream>>>(
            (__half *)output, (__half *)input, (__half *)vec1, (__half *)vec2, n, m,
            beta, alpha, stride1, stride2, out_stride_0, out_stride_1, in_stride_0,
            in_stride_1);
        break;
    case INFINI_DTYPE_BF16:
        Addr<__nv_bfloat16><<<grid, block, 0, stream>>>(
            (__nv_bfloat16 *)output, (__nv_bfloat16 *)input, (__nv_bfloat16 *)vec1,
            (__nv_bfloat16 *)vec2, n, m, beta, alpha, stride1, stride2,
            out_stride_0, out_stride_1, in_stride_0, in_stride_1);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *out, const void *input,
                                     const void *vec1, const void *vec2,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;

    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            _info.dtype, out, input, vec1, vec2, _info.vec1_size, _info.vec2_size,
            _info.beta, _info.alpha, _info.vec1_stride, _info.vec2_stride,
            _info.output_stride0, _info.output_stride1, _info.input_stride0,
            _info.input_stride1, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            _info.dtype, out, input, vec1, vec2, _info.vec1_size, _info.vec2_size,
            _info.beta, _info.alpha, _info.vec1_stride, _info.vec2_stride,
            _info.output_stride0, _info.output_stride1, _info.input_stride0,
            _info.input_stride1, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_2048>(
            _info.dtype, out, input, vec1, vec2, _info.vec1_size, _info.vec2_size,
            _info.beta, _info.alpha, _info.vec1_stride, _info.vec2_stride,
            _info.output_stride0, _info.output_stride1, _info.input_stride0,
            _info.input_stride1, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            _info.dtype, out, input, vec1, vec2, _info.vec1_size, _info.vec2_size,
            _info.beta, _info.alpha, _info.vec1_stride, _info.vec2_stride,
            _info.output_stride0, _info.output_stride1, _info.input_stride0,
            _info.input_stride1, stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::addr::nvidia
