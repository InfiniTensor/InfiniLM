#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "paged_caching_nvidia.cuh"

template <typename Tdata, int NUM_THREADS>
INFINIOP_CUDA_KERNEL pagedCaching(
    Tdata *k_cache, Tdata *v_cache,
    const Tdata *k, const Tdata *v,
    const int64_t *slot_mapping,
    const size_t head_size, const size_t block_size,
    const ptrdiff_t k_src_stride, const ptrdiff_t v_src_stride,
    const ptrdiff_t k_cache_block_stride, const ptrdiff_t v_cache_block_stride,
    const ptrdiff_t k_cache_head_stride, const ptrdiff_t v_cache_head_stride,
    const ptrdiff_t k_cache_slot_stride, const ptrdiff_t v_cache_slot_stride) {
    op::paged_caching::cuda::pagedCachingKernel<Tdata, NUM_THREADS>(
        k_cache, v_cache, k, v, slot_mapping, head_size,
        block_size, k_src_stride, v_src_stride,
        k_cache_block_stride, v_cache_block_stride, k_cache_head_stride, v_cache_head_stride, k_cache_slot_stride, v_cache_slot_stride);
}

namespace op::paged_caching::nvidia {
// PIMPL struct definition
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

// Destructor implementation
Descriptor::~Descriptor() {
    delete _opaque;
}

// Static factory method implementation
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slot_mapping_desc) {

    auto info = PagedCachingInfo::create(k_cache_desc, v_cache_desc, k_desc, v_desc, slot_mapping_desc);
    CHECK_RESULT(info);

    // Create and return the Descriptor instance.
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// The launchKernel function is a templated helper to encapsulate the CUDA kernel launch.
// It sets up grid/block dimensions and calls the device-side kernel.
template <int NUM_THREADS>
infiniStatus_t launchKernel(const PagedCachingInfo &info,
                            void *k_cache, void *v_cache,
                            infiniDtype_t dtype,
                            const void *k, const void *v,
                            const void *slot_mapping,
                            size_t num_tokens, size_t num_kv_heads, size_t head_size, size_t block_size,
                            ptrdiff_t k_src_stride, ptrdiff_t v_src_stride,
                            ptrdiff_t k_cache_block_stride, ptrdiff_t v_cache_block_stride,
                            ptrdiff_t k_cache_head_stride, ptrdiff_t v_cache_head_stride,
                            ptrdiff_t k_cache_slot_stride, ptrdiff_t v_cache_slot_stride,
                            cudaStream_t stream) {

    // Grid dimension is 1D, with one block per token, as we decided.
    dim3 grid(uint64_t(num_kv_heads), uint64_t(num_tokens), 1);
    // Block dimension is 1D, using the number of threads specified at compile time.
    dim3 block(NUM_THREADS);

    // This kernel does not require dynamic shared memory.
    size_t shared_mem_size = 0;

    // Launch the device-side CUDA kernel.
    if (dtype == INFINI_DTYPE_F16) {
        pagedCaching<half, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (half *)k_cache,
                (half *)v_cache,
                (const half *)k,
                (const half *)v,
                (const int64_t *)slot_mapping,
                head_size,
                block_size,
                k_src_stride,
                v_src_stride,
                k_cache_block_stride,
                v_cache_block_stride,
                k_cache_head_stride,
                v_cache_head_stride,
                k_cache_slot_stride,
                v_cache_slot_stride);
    } else if (dtype == INFINI_DTYPE_BF16) {
        pagedCaching<__nv_bfloat16, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (__nv_bfloat16 *)k_cache,
                (__nv_bfloat16 *)v_cache,
                (const __nv_bfloat16 *)k,
                (const __nv_bfloat16 *)v,
                (const int64_t *)slot_mapping,
                head_size,
                block_size,
                k_src_stride,
                v_src_stride,
                k_cache_block_stride,
                v_cache_block_stride,
                k_cache_head_stride,
                v_cache_head_stride,
                k_cache_slot_stride,
                v_cache_slot_stride);
    } else if (dtype == INFINI_DTYPE_F32) {
        pagedCaching<float, NUM_THREADS>
            <<<grid, block, shared_mem_size, stream>>>(
                (float *)k_cache,
                (float *)v_cache,
                (const float *)k,
                (const float *)v,
                (const int64_t *)slot_mapping,
                head_size,
                block_size,
                k_src_stride,
                v_src_stride,
                k_cache_block_stride,
                v_cache_block_stride,
                k_cache_head_stride,
                v_cache_head_stride,
                k_cache_slot_stride,
                v_cache_slot_stride);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

// Execution method implementation
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *k_cache, void *v_cache,
    const void *k, const void *v,
    const void *slot_mapping,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;

    // Dispatch logic based on the GPU's maximum threads per block.
    // This allows selecting the largest, most efficient block size the hardware supports.
    if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_1024) {
        // Dispatch based on data type for a 1024-thread block.
        launchKernel<CUDA_BLOCK_SIZE_1024>(
            _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
            _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.block_size,
            _info.k_src_stride, _info.v_src_stride,
            _info.k_cache_block_stride, _info.v_cache_block_stride,
            _info.k_cache_head_stride, _info.v_cache_head_stride,
            _info.k_cache_slot_stride, _info.v_cache_slot_stride,
            stream);
    } else if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_512) {
        launchKernel<CUDA_BLOCK_SIZE_512>(
            _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
            _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.block_size,
            _info.k_src_stride, _info.v_src_stride,
            _info.k_cache_block_stride, _info.v_cache_block_stride,
            _info.k_cache_head_stride, _info.v_cache_head_stride,
            _info.k_cache_slot_stride, _info.v_cache_slot_stride,
            stream);
    } else if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_4096) {
        launchKernel<CUDA_BLOCK_SIZE_4096>(
            _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
            _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.block_size,
            _info.k_src_stride, _info.v_src_stride,
            _info.k_cache_block_stride, _info.v_cache_block_stride,
            _info.k_cache_head_stride, _info.v_cache_head_stride,
            _info.k_cache_slot_stride, _info.v_cache_slot_stride,
            stream);
    } else {
        // If the GPU is older and supports fewer threads, return an error.
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::paged_caching::nvidia
