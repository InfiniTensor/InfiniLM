#ifndef __TAKE_MOORE_KERNEL_H__
#define __TAKE_MOORE_KERNEL_H__

#include <musa_bf16.h> // 必须包含，用于 __mt_bfloat16 定义
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <type_traits> // 用于 std::is_same_v

namespace op::take::moore {

typedef struct TakeOp {
public:
    // Take 算子涉及两种数据类型：数据本身 (T) 和 索引类型 (TIdx)
    template <typename T, typename TIdx>
    __device__ __forceinline__ void operator()(
        const size_t curr_idx,    // 当前线程处理的线性索引 (对应 output 和 indices 的位置)
        const size_t num_in,      // 输入张量的元素总数 (用于边界检查)
        const T *input_data,      // 输入数据指针
        const TIdx *indices_data, // 索引数据指针
        T *output_data            // 输出数据指针
    ) const {

        // 1. 获取当前位置的索引值
        // indices 和 output 形状一致，curr_idx 对应两者的扁平化索引
        TIdx idx = indices_data[curr_idx];

        // 2. 处理负索引 (支持 Python 风格的负数索引，如 -1 代表最后一个元素)
        // 如果业务不需要支持负索引，可以移除此逻辑，直接判断 idx < 0
        if (idx < 0) {
            idx += static_cast<TIdx>(num_in);
        }

        // 3. 边界检查与赋值
        if (idx >= 0 && idx < static_cast<TIdx>(num_in)) {
            // 合法索引：直接拷贝
            // half/bf16/float 在 MUSA 中通常支持直接赋值(=)，无需转换
            output_data[curr_idx] = input_data[idx];
        } else {
            // 越界处理：通常填充 0
            // 使用 constexpr 保持与 avg_pool 代码风格一致的类型处理
            if constexpr (std::is_same_v<T, half>) {
                output_data[curr_idx] = __float2half(0.0f);
            } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
                output_data[curr_idx] = __float2bfloat16(0.0f);
            } else {
                output_data[curr_idx] = static_cast<T>(0);
            }
        }
    }

} TakeOp;

} // namespace op::take::moore

#endif // __TAKE_MOORE_KERNEL_H__
