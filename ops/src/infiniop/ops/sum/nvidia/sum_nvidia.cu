#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "sum_nvidia.cuh"

namespace op::sum::nvidia {
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t *dim,
    size_t dim_size,
    bool keepdim) {
    auto result = SumInfo::create(output_desc, input_desc, dim, dim_size, keepdim);
    CHECK_RESULT(result);
    auto info = result.take();
    size_t workspace_size = 0;
    workspace_size += (input_desc->ndim() + output_desc->ndim()) * (sizeof(size_t) + sizeof(ptrdiff_t));
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <size_t BLOCK_SIZE, typename T>
infiniStatus_t launchKernel(
    const SumInfo &info,
    T *output, const T *input,
    cudaStream_t stream, void *workspace, size_t workspace_size) {
    size_t input_ndim = info.permuted_input_shape.size();
    size_t output_ndim = info.output_shape.size();
    size_t input_size = info.input_size;
    size_t output_size = info.output_size;
    size_t reduce_num = info.reduce_num;
    unsigned char *workspace_ptr = reinterpret_cast<unsigned char *>(workspace);
    size_t workspace_offset = 0;
    size_t *permuted_input_shape_cuda = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
    size_t *output_shape_cuda = permuted_input_shape_cuda + input_ndim;
    workspace_offset += (input_ndim + output_ndim) * sizeof(size_t);

    ptrdiff_t *permuted_input_strides_cuda = reinterpret_cast<ptrdiff_t *>(workspace_ptr + workspace_offset);
    ptrdiff_t *output_strides_cuda = permuted_input_strides_cuda + input_ndim;
    workspace_offset += (input_ndim + output_ndim) * sizeof(ptrdiff_t);

    CHECK_CUDA(cudaMemcpyAsync(permuted_input_shape_cuda, info.permuted_input_shape.data(), input_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_shape_cuda, info.output_shape.data(), output_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(permuted_input_strides_cuda, info.permuted_input_strides.data(), input_ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_strides_cuda, info.output_strides.data(), output_ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, stream));

    if (info.reduce_num == input_size) {
        T zero = static_cast<T>(0.0f);
        CHECK_CUDA(cudaMemcpyAsync(output, &zero, sizeof(T), cudaMemcpyHostToDevice, stream));
        size_t grid_size = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sumAllKernel<BLOCK_SIZE, T, T><<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(T), stream>>>(
            output, input, input_size, input_ndim, permuted_input_shape_cuda, permuted_input_strides_cuda);
    } else {
        size_t grid_size = (info.output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sumKernel<BLOCK_SIZE, T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            output, input, input_ndim, output_ndim, output_size, reduce_num,
            permuted_input_shape_cuda, output_shape_cuda, permuted_input_strides_cuda, output_strides_cuda);
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;

#define CALCULATE_SUM(BLOCK_SIZE, T)   \
    launchKernel<BLOCK_SIZE, T>(       \
        _info,                         \
        (T *)output, (const T *)input, \
        stream, workspace, workspace_size)

#define CALCULATE_SUM_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                        \
        if (_info.dtype == INFINI_DTYPE_BF16)                \
            return CALCULATE_SUM(BLOCK_SIZE, __nv_bfloat16); \
        else if (_info.dtype == INFINI_DTYPE_F16)            \
            return CALCULATE_SUM(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)            \
            return CALCULATE_SUM(BLOCK_SIZE, float);         \
        else                                                 \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;           \
    }

    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_SUM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_SUM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CALCULATE_SUM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_2048)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_SUM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::sum::nvidia
