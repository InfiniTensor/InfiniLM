#include <cmath>
#include <type_traits>

namespace op::cuda {

// SELU constants
constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;

struct SeluOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, float>) {
            return x > 0.0f ? SELU_SCALE * x : SELU_SCALE * SELU_ALPHA * (expf(x) - 1.0f);
        } else if constexpr (std::is_same_v<T, double>) {
            return x > 0.0 ? static_cast<double>(SELU_SCALE) * x : static_cast<double>(SELU_SCALE) * static_cast<double>(SELU_ALPHA) * (exp(x) - 1.0);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            const float xf = __bfloat162float(x);
            const float result = xf > 0.0f ? SELU_SCALE * xf : SELU_SCALE * SELU_ALPHA * (expf(xf) - 1.0f);
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, half>) {
            const float xf = __half2float(x);
            const float result = xf > 0.0f ? SELU_SCALE * xf : SELU_SCALE * SELU_ALPHA * (expf(xf) - 1.0f);
            return __float2half(result);
        } else {
            float xf = static_cast<float>(x);
            float result = xf > 0.0f ? SELU_SCALE * xf : SELU_SCALE * SELU_ALPHA * (expf(xf) - 1.0f);
            return static_cast<T>(result);
        }
    }
};

} // namespace op::cuda
