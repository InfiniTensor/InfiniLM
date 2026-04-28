#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "../cuda/kernel.cuh"
#include "var_mean_moore.h"

namespace op::var_mean::moore {
struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t var_output_desc,
    infiniopTensorDescriptor_t mean_output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t *dim,
    size_t dim_size,
    bool unbiased,
    bool keepdim) {
    auto result = VarMeanInfo::create(var_output_desc, input_desc, dim, dim_size, unbiased, keepdim);
    CHECK_RESULT(result);
    auto info = result.take();
    size_t workspace_size = 0;
    workspace_size += input_desc->ndim() * (sizeof(size_t) + sizeof(ptrdiff_t)); // permuted_input_shape + permuted_input_strides
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        info, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {
bool IsNanOut(const VarMeanInfo &info) {
    return (info.reduce_num == 0) || (info.reduce_num == 1 && info.unbiased_var == true);
}
template <size_t BLOCK_SIZE, typename Tdata, typename ComputeType>
infiniStatus_t launchKernel(
    const VarMeanInfo &info,
    Tdata *var_output, Tdata *mean_output, const Tdata *input,
    bool unbiased, bool keepdim,
    musaStream_t stream, void *workspace, size_t workspace_size) {
    size_t input_ndim = info.permuted_input_shape.size();
    size_t output_ndim = info.output_shape.size();
    size_t input_size = info.input_size;
    size_t output_size = info.output_size;
    size_t reduce_num = info.reduce_num;
    unsigned char *workspace_ptr = reinterpret_cast<unsigned char *>(workspace);
    size_t workspace_offset = 0;
    size_t *permuted_input_shape_musa = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
    workspace_offset += input_ndim * sizeof(size_t);

    ptrdiff_t *permuted_input_strides_musa = reinterpret_cast<ptrdiff_t *>(workspace_ptr + workspace_offset);
    workspace_offset += input_ndim * sizeof(ptrdiff_t);

    CHECK_MOORE(musaMemcpyAsync(permuted_input_shape_musa, info.permuted_input_shape.data(), input_ndim * sizeof(size_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(permuted_input_strides_musa, info.permuted_input_strides.data(), input_ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));
    bool is_nan = IsNanOut(info);
    if (info.reduce_num == input_size) { // scalar output
        ComputeType *tmp_buffer;
        constexpr size_t MAX_GRID_SIZE = 128;
        size_t grid_size = std::min(MAX_GRID_SIZE,
                                    (input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        grid_size = std::max(1UL, grid_size);
        CHECK_MOORE(musaMalloc(&tmp_buffer, grid_size * 3 * sizeof(ComputeType)));
        ComputeVarScalarOut<Tdata, ComputeType><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            input, var_output, mean_output, tmp_buffer, input_size, input_ndim,
            permuted_input_shape_musa, permuted_input_strides_musa, unbiased, is_nan);
        CHECK_MOORE(musaFree(tmp_buffer));
    } else {
        size_t grid_size = std::min(256UL, (info.output_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
        grid_size = std::max(1UL, grid_size);
        ComputeVarMeanUsingWelfordWrapper<Tdata, ComputeType><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            input, var_output, mean_output, input_ndim, output_size, reduce_num,
            permuted_input_shape_musa, permuted_input_strides_musa, unbiased, is_nan);
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *var_output,
    void *mean_output,
    const void *input,
    bool unbiased,
    bool keepdim,
    void *stream_) const {

    musaStream_t stream = (musaStream_t)stream_;

#define CALCULATE_VAR_MEAN(BLOCK_SIZE, Tdata, ComputeType)               \
    launchKernel<BLOCK_SIZE, Tdata, ComputeType>(                        \
        _info,                                                           \
        (Tdata *)var_output, (Tdata *)mean_output, (const Tdata *)input, \
        unbiased, keepdim,                                               \
        stream, workspace, workspace_size)

#define CALCULATE_VAR_MEAN_WITH_BLOCK_SIZE(BLOCK_SIZE)                    \
    {                                                                     \
        if (_info.dtype == INFINI_DTYPE_BF16)                             \
            return CALCULATE_VAR_MEAN(BLOCK_SIZE, __mt_bfloat16, double); \
        else if (_info.dtype == INFINI_DTYPE_F16)                         \
            return CALCULATE_VAR_MEAN(BLOCK_SIZE, half, double);          \
        else if (_info.dtype == INFINI_DTYPE_F32)                         \
            return CALCULATE_VAR_MEAN(BLOCK_SIZE, float, double);         \
        else                                                              \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                        \
    }

    if (_opaque->internal->maxThreadsPerBlock() >= 256) {
        CALCULATE_VAR_MEAN_WITH_BLOCK_SIZE(256)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::var_mean::moore
