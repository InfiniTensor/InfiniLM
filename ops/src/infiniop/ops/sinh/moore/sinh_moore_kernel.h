#ifndef __SINH_MOORE_KERNEL_H__
#define __SINH_MOORE_KERNEL_H__

#include <cmath>
#include <type_traits>

namespace op::sinh::moore {

typedef struct SinhOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float x0 = __low2float(x);
            float x1 = __high2float(x);
            return __floats2half2_rn(sinhf(x0), sinhf(x1));
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            return __float2half(sinhf(xf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(sinhf(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return sinhf(x);
        } else { // double
            return ::sinh(x);
        }
    }
} SinhOp;

} // namespace op::sinh::moore

#endif // __SINH_MOORE_KERNEL_H__
