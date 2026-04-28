#ifndef __EMBEDDING_CUDA_KERNEL_CUH__
#define __EMBEDDING_CUDA_KERNEL_CUH__

#include <type_traits>

// Helper function to check memory alignment
__forceinline__ __device__ bool is_aligned(const void *ptr, size_t alignment) {
    // Use size_t for pointer arithmetic in device code (more compatible)
    return (reinterpret_cast<size_t>(ptr) % alignment == 0);
}

// Vectorized copy for float type using float4
template <typename IndexType>
__forceinline__ __device__ void copyVectorizedFloat4(
    float *__restrict__ dst,
    const float *__restrict__ src,
    size_t embedding_dim) {
    // Use float4 for vectorized access (16 bytes, 4 floats)
    const float4 *src_vec = reinterpret_cast<const float4 *>(src);
    float4 *dst_vec = reinterpret_cast<float4 *>(dst);
    size_t vec_count = embedding_dim / 4;

    // Vectorized copy using __ldg for read-only weight
    for (size_t i = 0; i < vec_count; ++i) {
        dst_vec[i] = __ldg(&src_vec[i]);
    }

    // Copy remaining elements
    size_t remaining = embedding_dim % 4;
    if (remaining > 0) {
        size_t offset = vec_count * 4;
        for (size_t i = 0; i < remaining; ++i) {
            dst[offset + i] = __ldg(&src[offset + i]);
        }
    }
}

// Vectorized copy for float type using float2 (fallback when not aligned to 16 bytes)
template <typename IndexType>
__forceinline__ __device__ void copyVectorizedFloat2(
    float *__restrict__ dst,
    const float *__restrict__ src,
    size_t embedding_dim) {
    // Use float2 for vectorized access (8 bytes, 2 floats)
    const float2 *src_vec = reinterpret_cast<const float2 *>(src);
    float2 *dst_vec = reinterpret_cast<float2 *>(dst);
    size_t vec_count = embedding_dim / 2;

    // Vectorized copy using __ldg for read-only weight
    for (size_t i = 0; i < vec_count; ++i) {
        dst_vec[i] = __ldg(&src_vec[i]);
    }

    // Copy remaining element if odd
    if (embedding_dim % 2 != 0) {
        dst[embedding_dim - 1] = __ldg(&src[embedding_dim - 1]);
    }
}

// Vectorized copy for half type using half2
template <typename IndexType>
__forceinline__ __device__ void copyVectorizedHalf2(
    half *__restrict__ dst,
    const half *__restrict__ src,
    size_t embedding_dim) {
    // Use half2 for vectorized access (4 bytes, 2 halfs)
    const half2 *src_vec = reinterpret_cast<const half2 *>(src);
    half2 *dst_vec = reinterpret_cast<half2 *>(dst);
    size_t vec_count = embedding_dim / 2;

    // Vectorized copy using __ldg for read-only weight
    for (size_t i = 0; i < vec_count; ++i) {
        dst_vec[i] = __ldg(&src_vec[i]);
    }

    // Copy remaining element if odd
    if (embedding_dim % 2 != 0) {
        dst[embedding_dim - 1] = __ldg(&src[embedding_dim - 1]);
    }
}

// Vectorized copy for bfloat16 type using bfloat162
template <typename IndexType>
__forceinline__ __device__ void copyVectorizedBFloat162(
    cuda_bfloat16 *__restrict__ dst,
    const cuda_bfloat16 *__restrict__ src,
    size_t embedding_dim) {
    // Use bfloat162 for vectorized access (4 bytes, 2 bfloat16s)
    const cuda_bfloat162 *src_vec = reinterpret_cast<const cuda_bfloat162 *>(src);
    cuda_bfloat162 *dst_vec = reinterpret_cast<cuda_bfloat162 *>(dst);
    size_t vec_count = embedding_dim / 2;

    // Vectorized copy using __ldg for read-only weight
    for (size_t i = 0; i < vec_count; ++i) {
        dst_vec[i] = __ldg(&src_vec[i]);
    }

    // Copy remaining element if odd
    if (embedding_dim % 2 != 0) {
        dst[embedding_dim - 1] = __ldg(&src[embedding_dim - 1]);
    }
}

// Scalar copy fallback with __ldg optimization
template <typename T, typename IndexType>
__forceinline__ __device__ void copyScalar(
    T *__restrict__ dst,
    const T *__restrict__ src,
    size_t embedding_dim) {
    // Scalar copy with __ldg for read-only weight
    for (size_t i = 0; i < embedding_dim; ++i) {
        dst[i] = __ldg(&src[i]);
    }
}

#endif // __EMBEDDING_CUDA_KERNEL_CUH__
