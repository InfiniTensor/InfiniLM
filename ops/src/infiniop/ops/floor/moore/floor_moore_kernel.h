#ifndef __FLOOR_MOORE_KERNEL_H__
#define __FLOOR_MOORE_KERNEL_H__

/*
 * This file contains the Floor operation implementation for the MUSA backend.
 */

namespace op::floor::moore {
typedef struct FloorOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        if constexpr (std::is_same_v<T, half2>) {
            // MUSA 环境可能缺失 __h2floor，改用拆分转 float 处理
            // 提取低位和高位浮点数
            float f1 = __low2float(input);
            float f2 = __high2float(input);
            // 分别向下取整，然后合并回 half2
            // 使用 __floats2half2_rn (round-to-nearest) 进行转换合并
            return __floats2half2_rn(::floorf(f1), ::floorf(f2));
        } else if constexpr (std::is_same_v<T, half>) {
            // MUSA 环境缺失 __hfloor，改用转 float 处理
            return __float2half(::floorf(__half2float(input)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // Bfloat16 转 float 处理
            float val_f = __bfloat162float(input);
            return __float2bfloat16(::floorf(val_f));
        } else if constexpr (std::is_same_v<T, float>) {
            return ::floorf(input);
        } else {
            return ::floor(input);
        }
    }
} FloorOp;
} // namespace op::floor::moore

#endif // __FLOOR_MOORE_KERNEL_H__
