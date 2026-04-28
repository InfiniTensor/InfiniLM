#include <cmath>
#include <type_traits>

namespace op::cuda {

struct SinhOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, float>) {
            return sinhf(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return ::sinh(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            const float xf = __bfloat162float(x);
            return __float2bfloat16(sinhf(xf));
        } else if constexpr (std::is_same_v<T, half>) {
            const float xf = __half2float(x);
            return __float2half(sinhf(xf));
        } else {
            float xf = static_cast<float>(x);
            return static_cast<T>(sinhf(xf));
        }
    }
};

} // namespace op::cuda
