#ifndef __HARDTANH_CUDA_H__
#define __HARDTANH_CUDA_H__

#include <type_traits>

namespace op::hardtanh::cuda {

typedef struct HardTanhOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, float min_val, float max_val) const {
        if constexpr (std::is_same_v<T, half2>) {

            float2 x_f2 = __half22float2(x);
            x_f2.x = fminf(max_val, fmaxf(min_val, x_f2.x));
            x_f2.y = fminf(max_val, fmaxf(min_val, x_f2.y));
            return __float22half2_rn(x_f2);

        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {

            float x_f = __bfloat162float(x);
            return __float2bfloat16(fminf(max_val, fmaxf(min_val, x_f)));

        } else if constexpr (std::is_same_v<T, half>) {

            float x_f = __half2float(x);
            return __float2half(fminf(max_val, fmaxf(min_val, x_f)));

        } else if constexpr (std::is_same_v<T, float>) {

            return fminf(max_val, fmaxf(min_val, x));

        } else if constexpr (std::is_same_v<T, double>) {

            return fmin((double)max_val, fmax((double)min_val, x));
        }
    }
} HardTanhOp;

} // namespace op::hardtanh::cuda

#endif
