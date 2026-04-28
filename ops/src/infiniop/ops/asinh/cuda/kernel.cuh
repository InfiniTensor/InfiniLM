#ifndef __ASINH_CUDA_KERNEL_H__
#define __ASINH_CUDA_KERNEL_H__

namespace op::asinh::cuda {

typedef struct AsinhOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {

        if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x);
            return __float2half(asinhf(x_f));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float x_f = __bfloat162float(x);
            return __float2bfloat16(asinhf(x_f));
        } else if constexpr (std::is_same_v<T, float>) {
            return asinhf(x);
        } else {
            return ::asinh(x);
        }
    }

} AsinhOp;

} // namespace op::asinh::cuda

#endif // __ASINH_CUDA_KERNEL_H__
