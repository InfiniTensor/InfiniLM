#ifndef __FLOOR_CUDA_H__
#define __FLOOR_CUDA_H__

#include <cmath>
#include <type_traits>

namespace op::floor::cuda {

typedef struct FloorOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {

        // 1. Half2 (向量化)
        if constexpr (std::is_same_v<T, half2>) {
            float2 vf = __half22float2(x);
            float2 vr = make_float2(floorf(vf.x), floorf(vf.y));
            return __float22half2_rn(vr);
        }
        // 2. BFloat162 (向量化)
        else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(x));
            float f1 = __bfloat162float(__high2bfloat16(x));
            // 已修复：使用 _rn 标准函数
            return __floats2bfloat162_rn(floorf(f0), floorf(f1));
        }
        // 3. BFloat16 (标量)
        else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return __float2bfloat16(floorf(__bfloat162float(x)));
        }
        // 4. Half (标量)
        else if constexpr (std::is_same_v<T, half>) {
            return __float2half(floorf(__half2float(x)));
        }
        // 5. Float
        else if constexpr (std::is_same_v<T, float>) {
            return floorf(x);
        }
        // 6. Double
        else if constexpr (std::is_same_v<T, double>) {
            // 【关键修复】使用 ::floor 避免与 namespace op::floor 冲突
            return ::floor(x);
        }
        // 7. 整数
        else if constexpr (std::is_integral_v<T>) {
            return x;
        }
        // 8. 兜底
        else {
            // 【关键修复】使用 ::floor
            return ::floor(x);
        }
    }
} FloorOp;

} // namespace op::floor::cuda

#endif // __FLOOR_CUDA_H__
