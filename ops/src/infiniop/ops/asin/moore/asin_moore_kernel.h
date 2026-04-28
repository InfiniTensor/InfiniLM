#ifndef __ASIN_MOORE_KERNEL_H__
#define __ASIN_MOORE_KERNEL_H__

namespace op::asin::moore {
typedef struct AsinOp {
    static constexpr size_t num_inputs = 1;

    __device__ __forceinline__ float asin_f32_func(float x) const {
        return asinf(x);
    }
    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 vf = __half22float2(input);
            float2 vr = make_float2(asin_f32_func(vf.x), asin_f32_func(vf.y));
            return __float22half2_rn(vr);
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(input);
            float yf = asin_f32_func(xf);
            return __float2half_rn(yf);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(input));
            float f1 = __bfloat162float(__high2bfloat16(input));
            float r0 = asin_f32_func(f0);
            float r1 = asin_f32_func(f1);
            return __floats2bfloat162_rn(r0, r1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(input);
            float rf = asin_f32_func(xf);
            return __float2bfloat16_rn(rf);
        } else if constexpr (std::is_same_v<T, float>) {
            return asin_f32_func(input);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::asin(input);
        } else {
            return std::asin(input);
        }
    }
} AsinOp;
} // namespace op::asin::moore

#endif // __ASIN_MOORE_KERNEL_H__
