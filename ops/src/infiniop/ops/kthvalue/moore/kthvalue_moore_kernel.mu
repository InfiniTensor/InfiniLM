#include "../../../devices/moore/moore_handle.h"
#include "kthvalue_moore.h"
#include "kthvalue_moore_kernel.h"
#include <algorithm>
#include <cstdint>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::kthvalue::moore {

template <typename T>
static inline bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

static inline size_t next_power_of_2(size_t n) {
    if (n == 0) {
        return 1;
    }
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

template <typename T>
void launch_kernel(
    void *values,
    void *indices,
    const void *input,
    const KthvalueInfo &info,
    void *stream) {

    auto in_ptr = reinterpret_cast<const T *>(input);
    auto val_ptr = reinterpret_cast<T *>(values);
    auto idx_ptr = reinterpret_cast<int64_t *>(indices);

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    size_t dim_size = info.dim_size();
    size_t outer_size = info.outer_size();
    size_t inner_size = info.inner_size();
    int k = info.k();

    size_t power_of_2_dim = next_power_of_2(dim_size);

    size_t total_slices = outer_size * inner_size;

    unsigned int threads_per_block = std::max(1u, (unsigned int)(power_of_2_dim / 2));
    if (threads_per_block > 1024) {
        threads_per_block = 1024;
    }

    size_t smem_size = power_of_2_dim * sizeof(op::kthvalue::moore::KeyValuePair<T>);

    if (power_of_2_dim > 2048) {
        return;
    }

    op::kthvalue::moore::kthvalue_kernel<T>
        <<<total_slices, threads_per_block, smem_size, musa_stream>>>(
            val_ptr,
            idx_ptr,
            in_ptr,
            dim_size,
            inner_size,
            k,
            power_of_2_dim);
}

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t values_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t input_desc,
    int k,
    int dim,
    int keepdim) {

    auto info_result = KthvalueInfo::create(values_desc, indices_desc, input_desc, k, dim, keepdim);
    if (!info_result) {
        return info_result.status();
    }

    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(new Opaque(), info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *values,
    void *indices,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<__mt_bfloat16>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_I32:
        launch_kernel<int32_t>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_I64:
        launch_kernel<int64_t>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_U32:
        launch_kernel<uint32_t>(values, indices, input, _info, stream);
        break;
    case INFINI_DTYPE_U64:
        launch_kernel<uint64_t>(values, indices, input, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::kthvalue::moore
