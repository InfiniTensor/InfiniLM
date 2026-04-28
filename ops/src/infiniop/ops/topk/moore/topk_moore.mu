#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "../cuda/kernel.cuh"
#include "topk_moore.h"

#include <cub/block/block_radix_sort.cuh>
#include <cub/cub.cuh>

namespace op::topk::moore {
struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t values_output_desc,
    infiniopTensorDescriptor_t indices_output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t k,
    size_t dim,
    bool largest,
    bool sorted) {
    auto result = TopKInfo::create(values_output_desc, indices_output_desc, input_desc, k, dim, largest, sorted);
    CHECK_RESULT(result);
    auto info = result.take();
    size_t workspace_size = 0;

    workspace_size += (input_desc->ndim() + values_output_desc->ndim()) * (sizeof(size_t) + sizeof(ptrdiff_t));
    size_t dim_elements = input_desc->shape()[dim];
    size_t n_iteration = 1;
    for (size_t i = 0; i < input_desc->ndim(); i++) {
        if (i != dim) {
            n_iteration *= input_desc->shape()[i];
        }
    }
    size_t total = n_iteration * dim_elements;

    workspace_size += 3 * total * sizeof(uint32_t);
    workspace_size += 3 * total * sizeof(int32_t);
    workspace_size += n_iteration * k * (sizeof(uint32_t) + sizeof(int32_t));
    if (sorted) {
        workspace_size += n_iteration * k * (sizeof(uint32_t) + sizeof(int32_t));
    }
    workspace_size += 5 * n_iteration * sizeof(int32_t);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        info, workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <size_t BLOCK_SIZE, int32_t SORT_ITEMS_PER_THREAD, typename Tdata>
infiniStatus_t launchKernel(
    const TopKInfo &info,
    Tdata *values_output, int32_t *indices_output, const Tdata *input,
    size_t k, size_t dim, bool largest, bool sorted,
    musaStream_t stream, void *workspace, size_t workspace_size) {
    if (dim >= info.ndim) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (k == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    if (k > info.dim_elements) {
        return INFINI_STATUS_BAD_PARAM;
    }
    size_t input_ndim = info.ndim;
    size_t output_ndim = input_ndim;
    size_t n_iteration = info.n_iteration;
    size_t dim_elements = info.dim_elements;
    unsigned char *workspace_ptr = reinterpret_cast<unsigned char *>(workspace);
    size_t workspace_offset = 0;
    size_t *input_shape_musa = reinterpret_cast<size_t *>(workspace_ptr + workspace_offset);
    size_t *output_shape_musa = input_shape_musa + input_ndim;
    workspace_offset += (input_ndim + output_ndim) * sizeof(size_t);

    ptrdiff_t *input_strides_musa = reinterpret_cast<ptrdiff_t *>(workspace_ptr + workspace_offset);
    ptrdiff_t *output_strides_musa = input_strides_musa + input_ndim;
    workspace_offset += (input_ndim + output_ndim) * sizeof(ptrdiff_t);

    CHECK_MOORE(musaMemcpyAsync(input_shape_musa, info.input_shape.data(), input_ndim * sizeof(size_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(output_shape_musa, info.output_shape.data(), output_ndim * sizeof(size_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(input_strides_musa, info.input_strides.data(), input_ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));
    CHECK_MOORE(musaMemcpyAsync(output_strides_musa, info.output_strides.data(), output_ndim * sizeof(ptrdiff_t), musaMemcpyHostToDevice, stream));

    const int32_t total = n_iteration * dim_elements;

    uint32_t *cur_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(uint32_t);
    uint32_t *ones_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(uint32_t);
    uint32_t *zeros_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(uint32_t);

    int32_t *cur_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(int32_t);
    int32_t *ones_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(int32_t);
    int32_t *zeros_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += total * sizeof(int32_t);

    uint32_t *sel_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * k * sizeof(uint32_t);
    int32_t *sel_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * k * sizeof(int32_t);
    uint32_t *sel_sorted_vals = nullptr;
    int32_t *sel_sorted_idx = nullptr;
    if (sorted) {
        sel_sorted_vals = reinterpret_cast<uint32_t *>(workspace_ptr + workspace_offset);
        workspace_offset += n_iteration * k * sizeof(uint32_t);
        sel_sorted_idx = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
        workspace_offset += n_iteration * k * sizeof(int32_t);
    }

    int32_t *cur_n = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    int32_t *rem_k = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    int32_t *out_pos = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    int32_t *ones_count = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    int32_t *zeros_count = reinterpret_cast<int32_t *>(workspace_ptr + workspace_offset);
    workspace_offset += n_iteration * sizeof(int32_t);
    // init
    {
        size_t threads = 256;
        size_t blocks = (n_iteration + threads - 1) / threads;
        op::topk::cuda::init_row_state<<<blocks, threads, 0, stream>>>(cur_n, rem_k, out_pos, n_iteration, dim_elements, k);
    }
    // gather input -> cur
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((dim_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, n_iteration);
        op::topk::cuda::gather_rowwise<Tdata><<<grid, block, 0, stream>>>(
            input, cur_vals, cur_idx,
            n_iteration, dim_elements,
            input_ndim, dim,
            input_shape_musa, input_strides_musa);
    }
    // radix select/filter
    for (int bit = 31; bit >= 0; --bit) {
        {
            size_t threads = 256;
            size_t blocks = (n_iteration + threads - 1) / threads;
            op::topk::cuda::zero_row_counters<<<blocks, threads, 0, stream>>>(ones_count, zeros_count, n_iteration);
        }

        {
            dim3 block(BLOCK_SIZE);
            dim3 grid((dim_elements + BLOCK_SIZE - 1) / BLOCK_SIZE, n_iteration);
            op::topk::cuda::partition_rowwise<BLOCK_SIZE><<<grid, block, 0, stream>>>(
                cur_vals, cur_idx,
                ones_vals, ones_idx,
                zeros_vals, zeros_idx,
                cur_n, n_iteration, dim_elements,
                bit, largest,
                ones_count, zeros_count);
        }

        {
            op::topk::cuda::decide_and_compact<BLOCK_SIZE><<<n_iteration, BLOCK_SIZE, 0, stream>>>(
                cur_vals, cur_idx,
                ones_vals, ones_idx,
                zeros_vals, zeros_idx,
                ones_count, zeros_count,
                cur_n, rem_k, out_pos,
                sel_vals, sel_idx,
                n_iteration, dim_elements, k);
        }
    }

    // append remaining

    op::topk::cuda::take_remaining<BLOCK_SIZE><<<n_iteration, BLOCK_SIZE, 0, stream>>>(
        cur_vals, cur_idx,
        cur_n, rem_k, out_pos,
        sel_vals, sel_idx,
        n_iteration, dim_elements, k);

    // sort (CUB block radix sort)
    const int32_t *final_idx = sel_idx;

    if (sorted) {
        std::vector<int> h_offsets(n_iteration + 1);
        for (size_t i = 0; i <= n_iteration; i++) {
            h_offsets[i] = i * k;
        }
        int *d_offsets;
        CHECK_MOORE(musaMalloc(&d_offsets, (n_iteration + 1) * sizeof(int)));
        CHECK_MOORE(musaMemcpy(d_offsets, h_offsets.data(), (n_iteration + 1) * sizeof(int), musaMemcpyHostToDevice));

        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        if (!largest) {
            cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx,
                                                     n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
            musaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx,
                                                     n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
        } else {
            cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx,
                                                               n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
            musaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceSegmentedRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, sel_vals, sel_sorted_vals, sel_idx, sel_sorted_idx,
                                                               n_iteration * k, n_iteration, d_offsets, d_offsets + 1, 0, sizeof(uint32_t) * 8, stream);
        }
        CHECK_MOORE(musaFree(d_offsets));
        CHECK_MOORE(musaFree(d_temp_storage));
        final_idx = sel_sorted_idx;
    }

    // scatter to output (strided write)
    {
        dim3 block(BLOCK_SIZE);
        dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, n_iteration);
        op::topk::cuda::scatter_to_output<Tdata><<<grid, block, 0, stream>>>(
            input, final_idx,
            values_output, indices_output,
            n_iteration, k,
            input_ndim, dim,
            input_shape_musa, input_strides_musa,
            output_shape_musa, output_strides_musa);
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *values_output,
    void *indices_output,
    const void *input,
    size_t k,
    size_t dim,
    bool largest,
    bool sorted,
    void *stream_) const {

    musaStream_t stream = (musaStream_t)stream_;
    constexpr int ITEMS = 4;
#define CALCULATE_TOPK(BLOCK_SIZE, Tdata)                                        \
    launchKernel<BLOCK_SIZE, ITEMS, Tdata>(                                      \
        _info,                                                                   \
        (Tdata *)values_output, (int32_t *)indices_output, (const Tdata *)input, \
        k, dim, largest, sorted,                                                 \
        stream, workspace, workspace_size)

#define CALCULATE_TOPK_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                         \
        if (_info.dtype == INFINI_DTYPE_BF16)                 \
            return CALCULATE_TOPK(BLOCK_SIZE, __mt_bfloat16); \
        else if (_info.dtype == INFINI_DTYPE_F16)             \
            return CALCULATE_TOPK(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)             \
            return CALCULATE_TOPK(BLOCK_SIZE, float);         \
        else                                                  \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;            \
    }

    if (_opaque->internal->maxThreadsPerBlock() >= 256) {
        CALCULATE_TOPK_WITH_BLOCK_SIZE(256)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::topk::moore
