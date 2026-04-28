#ifndef __FLOOR_DIVIDE_CUDA_H__
#define __FLOOR_DIVIDE_CUDA_H__

#include <type_traits>

namespace op::floor_divide::cuda {
typedef struct FloorDivideOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2floor(__h2div(a, b));
        } else if constexpr (std::is_same_v<T, half>) {
            return hfloor(__hdiv(a, b));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float val = __bfloat162float(a) / __bfloat162float(b);
            return __float2bfloat16(floorf(val));
        } else if constexpr (std::is_same_v<T, float>) {
            return floorf(a / b);
        } else if constexpr (std::is_same_v<T, double>) {
            return floor(a / b);
        } else {
            T res = a / b;
            T rem = a % b;
            if (rem != 0 && ((a < 0) ^ (b < 0))) {
                res -= 1;
            }
            return res;
        }
    }
} FloorDivideOp;
} // namespace op::floor_divide::cuda

#endif // __FLOOR_DIVIDE_CUDA_H__
