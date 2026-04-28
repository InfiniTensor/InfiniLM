#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "../cuda/kernel.cuh"
#include "sum_moore.h"

namespace op::sum::moore {
struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
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
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        info, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <size_t BLOCK_SIZE, typename T>
infiniStatus_t launchKernel(
    const SumInfo &info,
    T *output, const T *input,
    musaStream_t stream, void *workspace, size_t workspace_size) {
    size_t input_ndim = info.permuted_input_shape.size();
    size_t output_ndim = info.output_shape.size();
    size_t input_size = info.input_size;
    size_t output_size = info.output_size;
    size_t reduce_num = info.reduce_num;
    unsigned char *workspace_ptr = reinterpret_cast<unsigned char *>(workspace);
    size_t workspace_offset = 0;
    size_t *permuted_input_shape_musa = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
    size_t *output_shape_musa = permuted_input_shape_musa + input_ndim;
    workspace_offset += (input_ndim + output_ndim) * sizeof(size_t);

    ptrdiff_t *permuted_input_strides_musa = reinterpret_cast<ptrdiff_t *>(workspace_ptr + workspace_offset);
    ptrdiff_t *output_strides_musa = permuted_input_strides_musa + input_ndim;
    workspace_offset += (input_ndim + output_ndim) * sizeof(ptrdiff_t);

    CHECK_MOORE(musaMemcpyAsync(permuted_input_shape_musa, info.permuted_input_shape.data(), input_ndim * sizeof(size_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(output_shape_musa, info.output_shape.data(), output_ndim * sizeof(size_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(output_strides_musa, info.output_strides.data(), output_ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(permuted_input_strides_musa, info.permuted_input_strides.data(), input_ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));

    if (info.reduce_num == input_size) {
        if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            // 需要解决 moore不支持bf16的atomic add的问题
            float zero = 0.0f;
            float *tmp_output;
            CHECK_MOORE(musaMalloc(&tmp_output, sizeof(float)));
            CHECK_MOORE(musaMemcpyAsync(tmp_output, &zero, sizeof(float), musaMemcpyHostToDevice, stream));
            size_t grid_size = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sumAllKernel<BLOCK_SIZE, T, float><<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float), stream>>>(
                tmp_output, input, input_size, input_ndim, permuted_input_shape_musa, permuted_input_strides_musa);
            // 可以自定义 kernel，将 float -> T，这里直接memcpy了
            float host_val;
            CHECK_MOORE(musaMemcpy(&host_val, tmp_output, sizeof(float), musaMemcpyDeviceToHost));
            T out_val = static_cast<T>(host_val);
            CHECK_MOORE(musaMemcpyAsync(output, &out_val, sizeof(T), musaMemcpyHostToDevice, stream));
            CHECK_MOORE(musaFree(tmp_output));
        } else {
            T zero = static_cast<T>(0.0f);
            CHECK_MOORE(musaMemcpyAsync(output, &zero, sizeof(T), musaMemcpyHostToDevice, stream));
            size_t grid_size = (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sumAllKernel<BLOCK_SIZE, T, T><<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(T), stream>>>(
                output, input, input_size, input_ndim, permuted_input_shape_musa, permuted_input_strides_musa);
        }
    } else {
        size_t grid_size = (info.output_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        sumKernel<BLOCK_SIZE, T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            output, input, input_ndim, output_ndim, output_size, reduce_num,
            permuted_input_shape_musa, output_shape_musa, permuted_input_strides_musa, output_strides_musa);
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

    musaStream_t stream = (musaStream_t)stream_;

#define CALCULATE_SUM(BLOCK_SIZE, T)   \
    launchKernel<BLOCK_SIZE, T>(       \
        _info,                         \
        (T *)output, (const T *)input, \
        stream, workspace, workspace_size)

#define CALCULATE_SUM_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                        \
        if (_info.dtype == INFINI_DTYPE_BF16)                \
            return CALCULATE_SUM(BLOCK_SIZE, __mt_bfloat16); \
        else if (_info.dtype == INFINI_DTYPE_F16)            \
            return CALCULATE_SUM(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)            \
            return CALCULATE_SUM(BLOCK_SIZE, float);         \
        else                                                 \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;           \
    }

    if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_1024) {
        CALCULATE_SUM_WITH_BLOCK_SIZE(MOORE_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_512) {
        CALCULATE_SUM_WITH_BLOCK_SIZE(MOORE_BLOCK_SIZE_512)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::sum::moore
