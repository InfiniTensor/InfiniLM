#ifndef __INDEX_COPY_CUDA_H__
#define __INDEX_COPY_CUDA_H__

#include <cstdint>

namespace op::index_copy::cuda {

// ==================================================================
// 1. 定义向量化数据包 (Aligned Pack)
// ==================================================================
// 与 IndexAdd 保持一致，用于向量化读取
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T val[N];
};

// ==================================================================
// 2. 标量版 Kernel (通用 fallback)
// ==================================================================
template <typename T, typename TIdx>
__global__ void index_copy_kernel(
    T *__restrict__ output,
    const T *__restrict__ source,
    const TIdx *__restrict__ indices,
    size_t outer_size, // dim 左边的维度积
    size_t inner_size, // dim 右边的维度积
    size_t dim_size,   // output 在 dim 维度的长度
    size_t index_len,  // index 的长度 (source 在 dim 维度的长度)
    size_t num_source  // source 的总元素数
    // 注意：移除了 float alpha
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

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

        // 4. 边界检查与赋值
        if (target_dim_idx >= 0 && target_dim_idx < static_cast<TIdx>(dim_size)) {
            // 计算 Output 的线性偏移
            // Output Shape: [Outer, DimSize, Inner]
            size_t out_offset = outer_idx * (dim_size * inner_size) + static_cast<size_t>(target_dim_idx) * inner_size + inner_idx;

            // 【核心修改】
            // IndexCopy 不需要原子操作，直接赋值。
            // 如果有多个索引指向同一个位置，结果由执行顺序决定（Race Condition），这是符合预期的行为。
            output[out_offset] = source[i];
        }
    }
}

// ==================================================================
// 3. 向量化 Kernel (优化读取带宽)
// ==================================================================
template <typename T, typename TIdx, int PackSize>
__global__ void index_copy_kernel_vectorized(
    T *__restrict__ output,
    const T *__restrict__ source,
    const TIdx *__restrict__ indices,
    size_t outer_size,
    size_t inner_size,
    size_t dim_size,
    size_t index_len,
    size_t num_packs // Source 的 Pack 数量
    // 注意：移除了 float alpha
) {
    // 将 source 强转为 Pack 指针，实现向量化读取
    using PackType = Pack<T, PackSize>;
    const PackType *src_vec = reinterpret_cast<const PackType *>(source);

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

            // 3. 赋值
            if (target_dim_idx >= 0 && target_dim_idx < static_cast<TIdx>(dim_size)) {
                size_t out_offset = outer_idx * (dim_size * inner_size) + static_cast<size_t>(target_dim_idx) * inner_size + inner_idx;

                // 【核心修改】直接赋值
                output[out_offset] = reg_pack.val[k];
            }
        }
    }
}

} // namespace op::index_copy::cuda

#endif // __INDEX_COPY_CUDA_H__
