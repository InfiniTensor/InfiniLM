#ifndef __SWIGLU_CUDA_H__
#define __SWIGLU_CUDA_H__

/*
 * This file contains the SwiGLU operation implementation for the MUSA backend.
 *
 * It uses the 'op::swiglu::cuda' namespace to maintain a consistent code structure
 * and interface with the CUDA implementation, ensuring code alignment across different
 * hardware platforms.
 */

namespace op::swiglu::cuda {
typedef struct SwiGLUOp {
private:
    template <typename T>
    __device__ __forceinline__ T sigmoid(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2rcp(__hadd2(make_half2(1, 1), h2exp(__hneg2(x))));
        } else if constexpr (std::is_same_v<T, half>) {
            // This implementation uses standard floating-point arithmetic to calculate the sigmoid function,
            // ensuring portability across on MUSA platforms.
            //
            // The original CUDA implementation's reliance on platform-specific intrinsics like hrcp for half-precision,
            // which was not supported on the MUSA platform.
            // To resolve this, the half-precision input is first converted to a higher-precision float,
            // the calculation is performed, and the result is cast back to half.
            float xf = __half2float(x);
            float sigf = 1.0f / (1.0f + std::exp(-xf));
            return __float2half(sigf);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float x0 = __bfloat162float(__low2bfloat16(x));
            float x1 = __bfloat162float(__high2bfloat16(x));
            float sig0 = __frcp_rn(__fadd_rn(1.0f, __expf(-x0)));
            float sig1 = __frcp_rn(__fadd_rn(1.0f, __expf(-x1)));
            return __floats2bfloat162_rn(sig0, sig1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(__frcp_rn(__fadd_rn(1.0f, __expf(-xf))));
        } else if constexpr (std::is_same_v<T, float>) {
            return __frcp_rn(__fadd_rn(1, __expf(-x)));
        } else {
            return 1 / (1 + std::exp(-x));
        }
    }

public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &up, const T &gate) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hmul2(__hmul2(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, half>) {
            return __hmul(__hmul(gate, sigmoid(gate)), up);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            cuda_bfloat162 sig = sigmoid(gate);

            // On the MUSA platform, `__low2float()` and `__high2float()` are used to directly
            // extract and convert bfloat16 values to float. These functions replace the
            // two-step process used in CUDA (e.g., `__low2bfloat16` followed by `__bfloat162float`).
            // Since MUSA may not support '__low2bfloat16'
            float gate0 = __low2float(gate);
            float gate1 = __high2float(gate);
            float sig0 = __low2float(sig);
            float sig1 = __high2float(sig);
            float up0 = __low2float(up);
            float up1 = __high2float(up);

            float res0 = __fmul_rn(__fmul_rn(gate0, sig0), up0);
            float res1 = __fmul_rn(__fmul_rn(gate1, sig1), up1);
            return __floats2bfloat162_rn(res0, res1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            cuda_bfloat16 sig = sigmoid(gate);
            float gatef = __bfloat162float(gate);
            float sigf = __bfloat162float(sig);
            float upf = __bfloat162float(up);
            return __float2bfloat16_rn(__fmul_rn(__fmul_rn(gatef, sigf), upf));
        } else if constexpr (std::is_same_v<T, float>) {
            return __fmul_rn(__fmul_rn(gate, sigmoid(gate)), up);
        } else {
            return gate * sigmoid(gate) * up;
        }
    }
} SwiGLUOp;
} // namespace op::swiglu::cuda

#endif // __SWIGLU_CUDA_H__
