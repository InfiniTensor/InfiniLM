#ifndef __HYPOT_MOORE_KERNEL_H__
#define __HYPOT_MOORE_KERNEL_H__

#include <cmath>
#include <musa_bf16.h>
#include <musa_fp16.h>

namespace op::hypot::moore {

typedef struct HypotOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const T &y) const {
        // -----------------------------------------------------------------
        // 1. Half2
        // -----------------------------------------------------------------
        if constexpr (std::is_same_v<T, half2>) {
            float x_low = __low2float(x);
            float x_high = __high2float(x);
            float y_low = __low2float(y);
            float y_high = __high2float(y);
            return __floats2half2_rn(::hypotf(x_low, y_low), ::hypotf(x_high, y_high));
        }
        // -----------------------------------------------------------------
        // 2. Half
        // -----------------------------------------------------------------
        else if constexpr (std::is_same_v<T, half>) {
            return __float2half(::hypotf(__half2float(x), __half2float(y)));
        }
        // -----------------------------------------------------------------
        // 3. Bfloat16 (__mt_bfloat16)
        // -----------------------------------------------------------------
        else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            // 将 __mt_bfloat16 转为 float 计算
            float x_f = __bfloat162float(x);
            float y_f = __bfloat162float(y);

            // 计算结果转回 __mt_bfloat16
            return __float2bfloat16(::hypotf(x_f, y_f));
        }
        // -----------------------------------------------------------------
        // 4. Float32
        // -----------------------------------------------------------------
        else if constexpr (std::is_same_v<T, float>) {
            return ::hypotf(x, y);
        }
        // -----------------------------------------------------------------
        // 5. Double / Other
        // -----------------------------------------------------------------
        else {
            return ::hypot(x, y);
        }
    }
} HypotOp;

} // namespace op::hypot::moore

#endif // __HYPOT_MOORE_KERNEL_H__
