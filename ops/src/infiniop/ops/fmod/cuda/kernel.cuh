#ifndef __FMOD_CUDA_H__
#define __FMOD_CUDA_H__

namespace op::fmod__::cuda {
typedef struct FmodOp {
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        // fmod(a, b) = a - b * trunc(a / b)
        if constexpr (std::is_same_v<T, half2>) {
            // 对于 half2，转换为 float 计算后再转回
            float2 af = __half22float2(a);
            float2 bf = __half22float2(b);
            float2 result;
            result.x = fmodf(af.x, bf.x);
            result.y = fmodf(af.y, bf.y);
            return __float22half2_rn(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            // 对于 bfloat162，转换为 float 计算后再转回
            float af_low = __bfloat162float(__low2bfloat16(a));
            float af_high = __bfloat162float(__high2bfloat16(a));
            float bf_low = __bfloat162float(__low2bfloat16(b));
            float bf_high = __bfloat162float(__high2bfloat16(b));
            return __floats2bfloat162_rn(fmodf(af_low, bf_low), fmodf(af_high, bf_high));
        } else if constexpr (std::is_same_v<T, half>) {
            // 对于 half，转换为 float 计算后再转回
            float af = __half2float(a);
            float bf = __half2float(b);
            return __float2half(fmodf(af, bf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // 对于 bfloat16，转换为 float 计算后再转回
            float af = __bfloat162float(a);
            float bf = __bfloat162float(b);
            return __float2bfloat16(fmodf(af, bf));
        } else if constexpr (std::is_same_v<T, float>) {
            return fmodf(a, b);
        } else if constexpr (std::is_same_v<T, double>) {
            return fmod(a, b);
        } else {
            // 整数类型使用 % 运算符
            return a % b;
        }
    }
} FmodOp;

} // namespace op::fmod__::cuda

#endif // __FMOD_CUDA_H__
