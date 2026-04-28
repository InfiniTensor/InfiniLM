#ifndef __FMIN_CUDA_H__
#define __FMIN_CUDA_H__

#include <type_traits>
namespace op::fmin::cuda {
typedef struct FminOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
#if defined(ENABLE_ILUVATAR_API)
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float a_f = __bfloat162float(a), b_f = __bfloat162float(b);
            return __float2bfloat16(fminf(a_f, b_f));
        } else if constexpr (std::is_same_v<T, half>) {
            float a_f = __half2float(a), b_f = __half2float(b);
            return __float2half(fminf(a_f, b_f));
        }
#elif defined(ENABLE_NVIDIA_API)
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return __hmin(a, b);
        }
#endif
        if constexpr (std::is_same_v<T, float>) {
            return fminf(a, b);
        } else {
            return a < b ? a : b;
        }
    }
} FminOp;
} // namespace op::fmin::cuda

#endif // __ADD_CUDA_H__
