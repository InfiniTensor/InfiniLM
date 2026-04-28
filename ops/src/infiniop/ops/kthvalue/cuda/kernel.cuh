#ifndef __KTHVALUE_CUDA_CUH__
#define __KTHVALUE_CUDA_CUH__

#include <cmath>
#include <cstdint>
#include <limits>

namespace op::kthvalue::cuda {

// ==================================================================
// 辅助结构: 键值对 (用于排序时携带索引)
// ==================================================================
template <typename T>
struct alignas(sizeof(int64_t) * 2) KeyValuePair { // 确保对齐
    T val;
    int64_t idx;

    __device__ __forceinline__ KeyValuePair() {}
    __device__ __forceinline__ KeyValuePair(T v, int64_t i) : val(v), idx(i) {}

    // 获取用于排序的“无穷大”值，用于 Padding
    __device__ __forceinline__ static KeyValuePair<T> max_value() {
        // 注意：这里需要根据 T 的具体类型返回最大值
        // 简单起见，对于浮点数我们使用 infinity，整数使用 max
        // 在实际工程中可能需要针对 half/bf16 的特化
        if constexpr (std::is_floating_point_v<T>) {
            return {static_cast<T>(INFINITY), -1};
        } else {
            // 简单的回退策略，实际可能需要 std::numeric_limits 的 device 版特化
            // 这里假设 T 支持强制转换 huge value
            return {static_cast<T>(1e30), -1};
        }
    }
};

// 针对 half/bf16 的比较辅助函数
// 如果系统头文件未重载 < 运算符，可能需要在此处添加
template <typename T>
__device__ __forceinline__ bool is_smaller(const T &a, const T &b) {
    return a < b;
}

// ==================================================================
// Bitonic Sort Helpers (Shared Memory)
// ==================================================================
template <typename T>
__device__ __forceinline__ void compare_and_swap(KeyValuePair<T> &a, KeyValuePair<T> &b, bool dir) {
    // dir: true for ascending, false for descending
    // 逻辑：如果 (a < b) != dir，说明顺序不对（或者 a > b 且 dir 为 true），则交换
    // 这里的 dir 含义：true 表示还需要保持 a < b

    // 自定义比较：先比值，值相同比索引（保持稳定性可选，这里简化为只比值）
    bool smaller = is_smaller(a.val, b.val) || (a.val == b.val && a.idx < b.idx);

    if (smaller != dir) {
        KeyValuePair<T> tmp = a;
        a = b;
        b = tmp;
    }
}

// ==================================================================
// Kernel: 基于 Bitonic Sort 的 KthValue
// ==================================================================
// 假设:
// 1. Grid 处理 Outer * Inner 个 Slice
// 2. 每个 Block 处理 1 个 Slice (Dim 维度)
// 3. Shared Memory 大小为 power_of_2_dim * sizeof(KeyValuePair)
// 4. BlockDim.x 至少为 power_of_2_dim / 2 (用于并行比较)
template <typename T>
__global__ void kthvalue_kernel(
    T *__restrict__ out_values,        // [Outer * Inner] (Flat)
    int64_t *__restrict__ out_indices, // [Outer * Inner] (Flat)
    const T *__restrict__ input,       // [Outer, Dim, Inner]
    size_t dim_size,
    size_t inner_size,
    int k,
    size_t power_of_2_dim // 扩展到 2 的幂次的大小
) {
    // 动态共享内存
    extern __shared__ char smem[];
    auto s_data = reinterpret_cast<KeyValuePair<T> *>(smem);

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    // 1. 计算当前 Slice 的基地址
    // Batch layout logic: flat_id -> (outer, inner)
    // 假设 GridDim.x = Outer * Inner
    size_t outer_idx = bid / inner_size;
    size_t inner_idx = bid % inner_size;

    // Input layout: [outer, dim, inner]
    // Base offset = outer * (dim_size * inner_size) + inner_idx
    // Stride = inner_size
    size_t input_base = outer_idx * dim_size * inner_size + inner_idx;
    size_t stride = inner_size;

    // 2. 加载数据到 Shared Memory (处理 Padding)
    // 循环加载，以支持 Dim > BlockDim 的情况 (虽然 Bitonic Sort 通常要求 threads >= N/2)
    for (unsigned int i = tid; i < power_of_2_dim; i += blockDim.x) {
        if (i < dim_size) {
            // 读取输入
            T val = input[input_base + i * stride];
            s_data[i] = KeyValuePair<T>(val, static_cast<int64_t>(i));
        } else {
            // Padding 最大值，使其排序后位于末尾
            s_data[i] = KeyValuePair<T>::max_value();
        }
    }
    __syncthreads();

    // 3. 双调排序 (Bitonic Sort)
    // 算法复杂度 O(log^2 N)
    for (unsigned int size = 2; size <= power_of_2_dim; size <<= 1) {
        // Bitonic Merge
        // dir: 升序或降序交替，构造双调序列
        // bool dir = (tid & (size / 2)) == 0;

        // 这里的逻辑稍微复杂，为了简单和稳定，我们使用全升序排序逻辑
        // 标准 Bitonic Sort 代码如下：

        for (unsigned int stride_step = size >> 1; stride_step > 0; stride_step >>= 1) {

            // 确保线程在范围内
            // 我们需要对所有 pairs (i, i+stride) 进行比较
            // 映射逻辑：
            // tid 0 处理: (0, stride), (2*stride, 3*stride)...
            // 这种映射较复杂，常用如下方式：
            // pos = 2*tid - (tid & (stride - 1)) ... 这种是 Butterfly 模式

            unsigned int pos = 2 * tid - (tid & (stride_step - 1));

            // 如果 pos + stride_step 在范围内
            if (pos + stride_step < power_of_2_dim) { // 边界检查，虽由 power_of_2_dim 保证
                unsigned int next_pos = pos + stride_step;

                // 计算比较方向
                // 在完整 Bitonic Sort 中，方向取决于 (pos & size)
                // 但这里我们仅实现简单的升序 Sort，
                // 需要更标准的 Bitonic Merge 网络:
                bool direction = ((pos & size) == 0);

                compare_and_swap(s_data[pos], s_data[next_pos], direction);
            }
            __syncthreads();
        }
    }

    // 4. 输出结果
    // 排序后，第 k 小的元素就在索引 k-1 处 (k is 1-based)
    if (tid == 0) {
        int target_k = k - 1;
        // 简单保护
        if (target_k >= 0 && target_k < dim_size) {
            out_values[bid] = s_data[target_k].val;
            out_indices[bid] = s_data[target_k].idx;
        } else {
            // Should not happen if validated
            // out_values[bid] = ...;
        }
    }
}

} // namespace op::kthvalue::cuda

#endif // __KTHVALUE_CUDA_CUH__
