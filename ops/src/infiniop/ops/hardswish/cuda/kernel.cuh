#ifndef __HARDSWISH_CUDA_H__
#define __HARDSWISH_CUDA_H__

#include <cmath>

namespace op::hardswish::cuda {

typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {

        if constexpr (std::is_same_v<T, half2>) {

            const half2 three = __float2half2_rn(3.0f);
            const half2 scale = __float2half2_rn(0.16666667f);

            half2 val = __hadd2(x, three);

#if defined(ENABLE_ILUVATAR_API)

            float2 val_f = __half22float2(val);
            val_f.x = fminf(fmaxf(val_f.x, 0.0f), 6.0f);
            val_f.y = fminf(fmaxf(val_f.y, 0.0f), 6.0f);
            val = __floats2half2_rn(val_f.x, val_f.y);
#else

            const half2 zero = __float2half2_rn(0.0f);
            const half2 six = __float2half2_rn(6.0f);

#if __CUDA_ARCH__ >= 800

            val = __hmin2(__hmax2(val, zero), six);
#else

            val = __hmax2(val, zero);
            val = __hmin2(val, six);
#endif
#endif

            return __hmul2(__hmul2(x, val), scale);

        }

        else if constexpr (std::is_same_v<T, cuda_bfloat16>) {

            const float x_f = __bfloat162float(x);

            const float val = fminf(fmaxf(x_f + 3.0f, 0.0f), 6.0f);
            return __float2bfloat16(x_f * val * 0.16666667f);

        }

        else if constexpr (std::is_same_v<T, half>) {
            const float x_f = __half2float(x);
            const float val = fminf(fmaxf(x_f + 3.0f, 0.0f), 6.0f);
            return __float2half(x_f * val * 0.16666667f);

        }

        else if constexpr (std::is_same_v<T, float>) {

            const float val = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
            return x * val * 0.16666667f;

        }

        else if constexpr (std::is_same_v<T, double>) {
            const double val = fmin(fmax(x + 3.0, 0.0), 6.0);
            return x * val * (1.0 / 6.0);
        }
    }
} HardSwishOp;

} // namespace op::hardswish::cuda

#endif
