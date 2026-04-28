#ifndef __ADDCMUL_METAX_KERNEL_H__
#define __ADDCMUL_METAX_KERNEL_H__

/*
 * This file contains the Addcmul operation implementation for the MUSA backend.
 * Formula: out = input + value * tensor1 * tensor2
 */

namespace op::addcmul::metax {

typedef struct AddcmulOp {
public:
    // 三元算子，输入为 input, tensor1, tensor2
    static constexpr size_t num_inputs = 3;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &in, const T &t1, const T &t2, float value) const {
        if constexpr (std::is_same_v<T, float>) {
            // F32 直接使用乘加指令
            return in + value * t1 * t2;
        } else if constexpr (std::is_same_v<T, half>) {
            // F16 提升到 float 计算以防止中间乘法溢出
            float f_in = __half2float(in);
            float f_t1 = __half2float(t1);
            float f_t2 = __half2float(t2);
            return __float2half(f_in + value * f_t1 * f_t2);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16 同样提升到 float 计算
            float f_in = __bfloat162float(in);
            float f_t1 = __bfloat162float(t1);
            float f_t2 = __bfloat162float(t2);
            return __float2bfloat16_rn(f_in + value * f_t1 * f_t2);
        } else if constexpr (std::is_same_v<T, double>) {
            return in + (double)value * t1 * t2;
        } else {
            // 整数类型或其他类型
            return in + static_cast<T>(value) * t1 * t2;
        }
    }
} AddcmulOp;

} // namespace op::addcmul::metax

#endif // __ADDCMUL_METAX_KERNEL_H__
