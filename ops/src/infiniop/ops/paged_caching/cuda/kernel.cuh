#ifndef __PAGED_CACHING_KERNEL_CUH__
#define __PAGED_CACHING_KERNEL_CUH__

//================================================================================
// Paged Caching Operator CUDA Kernel
//
// This kernel implements the "paged_caching" operation, which copies Key and Value
// vectors from a contiguous source tensor into a paged, non-contiguous KV Cache.
//
// Design Principles:
// 1. Token-Centric Parallelism: A 1D grid of `num_tokens` is launched. Each CUDA
//    block is responsible for caching one full token (all its heads).
// 2. Coalesced Memory Access: This grid strategy ensures that threads within a
//    block read a large, contiguous chunk of memory from the source tensors,
//    maximizing memory bandwidth utilization.
// 3. Vectorization: The copy operation is vectorized to further enhance memory
//    throughput, processing multiple data elements in a single instruction.
//================================================================================

namespace op::paged_caching::cuda {

template <
    typename Tdata, // Data type of the tensors (e.g., half, __nv_bfloat16)
    int NUM_THREADS // Number of threads per block, configured at launch time
    >
__device__ void pagedCachingKernel(
    // ----- Output Tensors -----
    Tdata *k_cache_ptr, // Pointer to the destination K cache pool [num_blocks, nkvh, block_size, dh]
    Tdata *v_cache_ptr, // Pointer to the destination V cache pool [num_blocks, nkvh, block_size, dh]
    // ----- Input Tensors -----
    const Tdata *k_ptr,              // Pointer to the source Keys, shape [ntok, nkvh, dh]
    const Tdata *v_ptr,              // Pointer to the source Values, shape [ntok, nkvh, dh]
    const int64_t *slot_mapping_ptr, // Pointer to the slot mapping, shape [ntok]
    // ----- Metadata -----
    const size_t head_size,  // Dimension of each head (dh)
    const size_t block_size, // Number of tokens per block in the KV cache
    // ----- Stride Information -----
    const ptrdiff_t k_src_stride,         // Stride between tokens in the source K tensor
    const ptrdiff_t v_src_stride,         // Stride between tokens in the source V tensor
    const ptrdiff_t k_cache_block_stride, // Stride between blocks in the K cache pool
    const ptrdiff_t v_cache_block_stride, // Stride between blocks in the V cache pool
    const ptrdiff_t k_cache_head_stride,  // Stride between heads in the K cache pool
    const ptrdiff_t v_cache_head_stride,  // Stride between heads in the V cache pool
    const ptrdiff_t k_cache_slot_stride,  // Stride between block slots in the K cache pool
    const ptrdiff_t v_cache_slot_stride   // Stride between block slots in the V cache pool
) {
    //================================================================================
    // 1. Identify Work Unit & Calculate Addresses
    //================================================================================

    // Each block processes one token.
    const int token_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    // const int num_kv_heads = gridDim.y;

    // Retrieve the destination slot for the current token.
    const int64_t slot_idx = slot_mapping_ptr[token_idx];

    // Handle padding: if slot_idx is negative, this token is padding and should be ignored.
    if (slot_idx < 0) {
        return;
    }
    // Calculate the physical block index and the offset within that block.
    const int64_t physical_block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    // Calculate base pointers for source and destination for this specific token.
    const Tdata *k_src_head_ptr = k_ptr + token_idx * k_src_stride + head_idx * head_size;
    const Tdata *v_src_head_ptr = v_ptr + token_idx * v_src_stride + head_idx * head_size;

    // Destination pointer calculation assumes a [num_blocks, block_size, num_heads, head_size] layout.
    // We point to the beginning of the memory region for this token's slot.
    Tdata *k_cache_block_base_ptr = k_cache_ptr + physical_block_idx * k_cache_block_stride;
    Tdata *k_dst_head_ptr = k_cache_block_base_ptr + head_idx * k_cache_head_stride + block_offset * k_cache_slot_stride;

    Tdata *v_cache_block_base_ptr = v_cache_ptr + physical_block_idx * v_cache_block_stride;
    Tdata *v_dst_head_ptr = v_cache_block_base_ptr + head_idx * v_cache_head_stride + block_offset * v_cache_slot_stride;

    //================================================================================
    // 2. Perform Element-wise Data Copy (Safe, Non-Vectorized)
    //================================================================================
    for (int i = threadIdx.x; i < head_size; i += NUM_THREADS) {
        k_dst_head_ptr[i] = k_src_head_ptr[i];
        v_dst_head_ptr[i] = v_src_head_ptr[i];
    }
}

} // namespace op::paged_caching::cuda

#endif // __PAGED_CACHING_KERNEL_CUH__
