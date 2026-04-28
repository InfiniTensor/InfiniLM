#ifndef __INDEX_COPY_MOORE_KERNEL_H__
#define __INDEX_COPY_MOORE_KERNEL_H__

#include <musa_bf16.h> // 必须包含，用于 __mt_bfloat16 定义
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <type_traits> // 用于 std::is_same_v

namespace op::index_copy::moore {

typedef struct IndexCopyOp {
public:
    // IndexCopy 算子涉及两种数据类型：数据本身 (T) 和 索引类型 (TIdx)
    // 逻辑：Output[..., indices[i], ...] = Source[..., i, ...]

    template <typename T, typename TIdx>
    __device__ __forceinline__ void operator()(
        const size_t curr_idx, // 当前线程处理的 Source 元素的线性索引

        // 几何参数 (来自 IndexCopyInfo)
        const size_t index_len,  // Index 向量的长度 (即 Source 在 dim 维度的长度)
        const size_t inner_size, // dim 右侧维度的 stride
        const size_t dim_size,   // Output 在 dim 维度的长度 (用于计算偏移和边界检查)

        // 指针
        const T *source_data,     // Source 数据指针 (Input source)
        const TIdx *indices_data, // 索引数据指针
        T *output_data            // Output 数据指针 (In/Out, 通常已由 Input 初始化)
    ) const {

        // 1. 坐标映射 (Flat Index -> Multi-dim Index)
        // Source 逻辑形状为: [Outer, IndexLen, Inner]
        // 将 curr_idx 分解为 (outer, idx_in_index, inner)

        size_t inner_idx = curr_idx % inner_size;
        size_t tmp = curr_idx / inner_size;
        size_t idx_in_indices = tmp % index_len; // 当前处理的是 indices 张量中的第几个索引
        size_t outer_idx = tmp / index_len;

        // 2. 读取 Source 数据
        T src_val = source_data[curr_idx];

        // 3. 获取目标 Index
        TIdx target_dim_idx = indices_data[idx_in_indices];

        // 4. 处理负索引 (支持 Python 风格，如 -1 代表最后一个元素)
        if (target_dim_idx < 0) {
            target_dim_idx += static_cast<TIdx>(dim_size);
        }

        // 5. 边界检查与赋值
        if (target_dim_idx >= 0 && target_dim_idx < static_cast<TIdx>(dim_size)) {
            // 计算 Output 的线性偏移
            // Output 逻辑形状: [Outer, DimSize, Inner]
            // Offset = Outer_Pos + Dim_Pos + Inner_Pos
            size_t out_offset = outer_idx * (dim_size * inner_size) + static_cast<size_t>(target_dim_idx) * inner_size + inner_idx;

            // 直接赋值 (Scatter)
            // MUSA 平台 half/bf16/float 均支持 operator=
            // 对于重复索引的情况，通常行为是非确定性的 (Last write wins)，无需原子操作
            output_data[out_offset] = src_val;
        }
        // 如果索引越界，IndexCopy 通常忽略该操作，不做任何修改
    }

} IndexCopyOp;

} // namespace op::index_copy::moore

#endif // __INDEX_COPY_MOORE_KERNEL_H__
