#ifndef __LOG1P_KERNEL_CUH__
#define __LOG1P_KERNEL_CUH__
#include <cmath>
#include <type_traits>

namespace op::cuda {

struct Log1pOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, float>) {
            // Use double precision for better accuracy.
            return (float)::log1p((double)x);
        } else if constexpr (std::is_same_v<T, double>) {
            return ::log1p(x);
        } else {
            // For F16/BF16: promote to float, compute, then cast back.
            return (T)(float)::log1p((double)(float)x);
        }
    }
};

} // namespace op::cuda
#endif
