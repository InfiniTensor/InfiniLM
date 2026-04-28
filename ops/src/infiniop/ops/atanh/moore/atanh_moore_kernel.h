#ifndef __ATANH_MOORE_KERNEL_H__
#define __ATANH_MOORE_KERNEL_H__

/*
 * This file contains the Atanh operation implementation for the MUSA backend.
 *
 * It follows the consistent code structure to ensure alignment across different
 * hardware platforms within the Moore Threads (MUSA) ecosystem.
 */
namespace op::atanh::moore {

typedef struct AtanhOp {
public:
    // 一元算子，输入数量为 1
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, half2>) {
            // 针对 half2 进行并行计算
            float2 f2 = __half22float2(a);
            f2.x = atanhf(f2.x);
            f2.y = atanhf(f2.y);
            return __float22half2_rn(f2);
        } else if constexpr (std::is_same_v<T, half>) {
            // 转为 float 计算以保证精度并匹配 MUSA 数学库
            return __float2half(atanhf(__half2float(a)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16 同样提升到 float 计算，避免转换歧义
            float a_f = __bfloat162float(a);
            return __float2bfloat16_rn(atanhf(a_f));
        } else if constexpr (std::is_same_v<T, float>) {
            // 调用 MUSA 内置的单精度反双曲正切函数
            return atanhf(a);
        } else if constexpr (std::is_same_v<T, double>) {
            // 调用双精度版本
            return ::atanh(a);
        } else {
            // 兜底实现（如果是整数类型，通常会隐式转为 float）
            return static_cast<T>(atanhf(static_cast<float>(a)));
        }
    }
} AtanhOp;

} // namespace op::atanh::moore

#endif // __ATANH_MOORE_KERNEL_H__
