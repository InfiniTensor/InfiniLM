#ifndef __RECIPROCAL_MOORE_KERNEL_H__
#define __RECIPROCAL_MOORE_KERNEL_H__

/*
 * This file contains the Reciprocal operation implementation for the MUSA backend.
 *
 * It follows the consistent code structure to ensure alignment across different
 * hardware platforms within the Moore Threads (MUSA) ecosystem.
 */
namespace op::reciprocal::moore {

typedef struct ReciprocalOp {
public:
    // 一元算子，输入数量为 1
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a) const {
        if constexpr (std::is_same_v<T, half2>) {
            // 使用 MUSA 的 half2 倒数指令（如果硬件支持）
            // 或者转为 float2 进行计算
            float2 f2 = __half22float2(a);
            f2.x = 1.0f / f2.x;
            f2.y = 1.0f / f2.y;
            return __float22half2_rn(f2);
        } else if constexpr (std::is_same_v<T, half>) {
            // 提升到 float 计算以保证数值稳定性
            return __float2half(1.0f / __half2float(a));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16 在 MUSA 上推荐转为 float 处理
            float a_f = __bfloat162float(a);
            return __float2bfloat16_rn(1.0f / a_f);
        } else if constexpr (std::is_same_v<T, float>) {
            // 编译器通常会将 1.0f/a 优化为硬件 rcp 指令 (Round to Nearest)
            return 1.0f / a;
        } else if constexpr (std::is_same_v<T, double>) {
            return 1.0 / a;
        } else {
            // 整数类型倒数通常返回 0 (除 1 以外)，保持标准 C++ 行为
            return static_cast<T>(1) / a;
        }
    }
} ReciprocalOp;

} // namespace op::reciprocal::moore

#endif // __RECIPROCAL_MOORE_KERNEL_H__
