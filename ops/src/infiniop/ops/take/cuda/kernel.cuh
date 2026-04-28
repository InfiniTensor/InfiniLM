#ifndef __TAKE_CUDA_H__
#define __TAKE_CUDA_H__

#include <cstdint>

namespace op::take::cuda {

// ==================================================================
// 1. 定义向量化数据包 (Aligned Pack)
// ==================================================================
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T val[N];
};

// ==================================================================
// 2. 标量版 Kernel (用于处理非对齐数据或尾部剩余数据)
// ==================================================================
template <typename T, typename TIdx>
__global__ void take_kernel(
    T *__restrict__ output,
    const T *__restrict__ input,
    const TIdx *__restrict__ indices,
    size_t num_out,
    size_t num_in) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < num_out; i += stride) {
        TIdx idx = __ldg(&indices[i]);

        // 标量读取
        if (idx >= 0 && idx < static_cast<TIdx>(num_in)) {
            output[i] = input[idx];
        } else {
            output[i] = static_cast<T>(0);
        }
    }
}

// ==================================================================
// 3. 向量化 Kernel (优化版)
// ==================================================================
/**
 * @tparam PackSize 每个线程处理的元素个数 (目标是凑齐 128-bit, e.g., float x 4)
 */
template <typename T, typename TIdx, int PackSize>
__global__ void take_kernel_vectorized(
    T *__restrict__ output,
    const T *__restrict__ input,
    const TIdx *__restrict__ indices,
    size_t num_packs, // 需要处理的 Pack 数量 (num_out / PackSize)
    size_t num_in) {
    // 将 output 强转为 Pack 指针，实现向量化写入
    using PackType = Pack<T, PackSize>;
    PackType *out_vec = reinterpret_cast<PackType *>(output);

    // Grid-Stride Loop 遍历 Pack
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < num_packs; i += stride) {
        PackType reg_pack;              // 寄存器缓存
        size_t base_idx = i * PackSize; // 当前 Pack 对应的原始 output 起始索引

// 循环展开 (Unroll): 关键优化点
// 编译器会展开这个循环，生成 PackSize 个独立的 Load 指令
// 从而利用 ILP (Instruction Level Parallelism) 掩盖 Input 的随机读取延迟
#pragma unroll
        for (int k = 0; k < PackSize; ++k) {
            // 读取索引 (Indices 是连续的，L1 Cache 命中率高)
            // 注意：Indices 类型大小可能与 T 不同，所以独立读取
            TIdx gather_idx = indices[base_idx + k];

            // 随机读取 (Gather)
            if (gather_idx >= 0 && gather_idx < static_cast<TIdx>(num_in)) {
                reg_pack.val[k] = input[gather_idx];
            } else {
                reg_pack.val[k] = static_cast<T>(0);
            }
        }

        // 向量化写入 (一次 STG.128 指令)
        out_vec[i] = reg_pack;
    }
}

} // namespace op::take::cuda

#endif // __TAKE_CUDA_H__
