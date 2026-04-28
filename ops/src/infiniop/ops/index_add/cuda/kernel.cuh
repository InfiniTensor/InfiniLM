#ifndef __INDEX_ADD_CUDA_H__
#define __INDEX_ADD_CUDA_H__

#include <cstdint>

namespace op::index_add::cuda {

__device__ __forceinline__ void atomic_add_custom(__half *address, __half val) {
#if __CUDA_ARCH__ >= 700
    atomicAdd(address, val);
#else
    // Fallback for older architectures (< Volta)
    unsigned int *address_as_ui = (unsigned int *)((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short old_val_raw = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        __half old_val = *reinterpret_cast<__half *>(&old_val_raw);

        __half new_val = old_val + val;
        unsigned short new_val_raw = *reinterpret_cast<unsigned short *>(&new_val);

        unsigned int new_int = (size_t)address & 2 ? (old & 0xffff) | (new_val_raw << 16)
                                                   : (old & 0xffff0000) | new_val_raw;

        old = atomicCAS(address_as_ui, assumed, new_int);
    } while (assumed != old);
#endif
}

__device__ __forceinline__ void atomic_add_custom(cuda_bfloat16 *address, cuda_bfloat16 val) {
#if __CUDA_ARCH__ >= 800
    atomicAdd(address, val);
#else
    // Fallback for older architectures (< Ampere)
    unsigned int *address_as_ui = (unsigned int *)((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short old_val_raw = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        cuda_bfloat16 old_val = *reinterpret_cast<cuda_bfloat16 *>(&old_val_raw);

        cuda_bfloat16 new_val = old_val + val;
        unsigned short new_val_raw = *reinterpret_cast<unsigned short *>(&new_val);

        unsigned int new_int = (size_t)address & 2 ? (old & 0xffff) | (new_val_raw << 16)
                                                   : (old & 0xffff0000) | new_val_raw;

        old = atomicCAS(address_as_ui, assumed, new_int);
    } while (assumed != old);
#endif
}

__device__ __forceinline__ void atomic_add_custom(int64_t *address, int64_t val) {
    atomicAdd(reinterpret_cast<unsigned long long int *>(address), static_cast<unsigned long long int>(val));
}

// --- 通用模板 (float, double, int32 等) ---
template <typename T>
__device__ __forceinline__ void atomic_add_custom(T *address, T val) {
    atomicAdd(address, val);
}

// ==================================================================
// 2. 定义向量化数据包 (Aligned Pack)
// ==================================================================
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T val[N];
};

// ==================================================================
// 3. 标量版 Kernel (通用 fallback)
// ==================================================================
template <typename T, typename TIdx>
__global__ void index_add_kernel(
    T *__restrict__ output,
    const T *__restrict__ source,
    const TIdx *__restrict__ indices,
    size_t outer_size, // dim 左边的维度积
    size_t inner_size, // dim 右边的维度积
    size_t dim_size,   // output 在 dim 维度的长度
    size_t index_len,  // index 的长度 (source 在 dim 维度的长度)
    size_t num_source, // source 的总元素数
    float alpha        // 缩放因子
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    T alpha_val = static_cast<T>(alpha);

    // Grid-Stride Loop 遍历 Source 张量
    for (size_t i = tid; i < num_source; i += stride) {
        // 1. 将线性索引 i 转换为逻辑坐标 (outer, idx_idx, inner)
        // Source Shape: [Outer, IndexLen, Inner]
        size_t inner_idx = i % inner_size;
        size_t tmp = i / inner_size;
        size_t idx_idx = tmp % index_len;
        size_t outer_idx = tmp / index_len;

        // 2. 读取索引值
        TIdx target_dim_idx = indices[idx_idx];

        // 3. 处理负索引 (防御性)
        if (target_dim_idx < 0) {
            target_dim_idx += static_cast<TIdx>(dim_size);
        }

        // 4. 边界检查与原子累加
        if (target_dim_idx >= 0 && target_dim_idx < static_cast<TIdx>(dim_size)) {
            // 计算 Output 的线性偏移
            // Output Shape: [Outer, DimSize, Inner]
            size_t out_offset = outer_idx * (dim_size * inner_size) + static_cast<size_t>(target_dim_idx) * inner_size + inner_idx;

            // 使用自定义原子操作
            atomic_add_custom(&output[out_offset], source[i] * alpha_val);
        }
    }
}

// ==================================================================
// 4. 向量化 Kernel (优化读取带宽)
// ==================================================================
template <typename T, typename TIdx, int PackSize>
__global__ void index_add_kernel_vectorized(
    T *__restrict__ output,
    const T *__restrict__ source,
    const TIdx *__restrict__ indices,
    size_t outer_size,
    size_t inner_size,
    size_t dim_size,
    size_t index_len,
    size_t num_packs, // Source 的 Pack 数量
    float alpha) {
    // 将 source 强转为 Pack 指针，实现向量化读取
    using PackType = Pack<T, PackSize>;
    const PackType *src_vec = reinterpret_cast<const PackType *>(source);

    T alpha_val = static_cast<T>(alpha);

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < num_packs; i += stride) {
        // 向量化读取 (LDG.128)
        PackType reg_pack = src_vec[i];

        // 当前 Pack 在 Source 中的起始线性索引
        size_t base_idx = i * PackSize;

// 循环展开：处理 Pack 中的每一个元素
#pragma unroll
        for (int k = 0; k < PackSize; ++k) {
            size_t curr_src_idx = base_idx + k;

            // 1. 坐标变换
            size_t inner_idx = curr_src_idx % inner_size;
            size_t tmp = curr_src_idx / inner_size;
            size_t idx_idx = tmp % index_len;
            size_t outer_idx = tmp / index_len;

            // 2. 读取 Index
            TIdx target_dim_idx = indices[idx_idx];

            if (target_dim_idx < 0) {
                target_dim_idx += static_cast<TIdx>(dim_size);
            }

            // 3. 原子累加
            if (target_dim_idx >= 0 && target_dim_idx < static_cast<TIdx>(dim_size)) {
                size_t out_offset = outer_idx * (dim_size * inner_size) + static_cast<size_t>(target_dim_idx) * inner_size + inner_idx;

                // 使用自定义原子操作
                atomic_add_custom(&output[out_offset], reg_pack.val[k] * alpha_val);
            }
        }
    }
}

} // namespace op::index_add::cuda

#endif // __INDEX_ADD_CUDA_H__
