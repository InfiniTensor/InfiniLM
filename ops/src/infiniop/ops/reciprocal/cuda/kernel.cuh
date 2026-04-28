#ifndef __RECIPROCAL_CUDA_H__
#define __RECIPROCAL_CUDA_H__

namespace op::reciprocal::cuda {
typedef struct ReciprocalOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            return h2rcp(x);
        } else if constexpr (std::is_same_v<T, half>) {
            return hrcp(x);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // bfloat16 does not have a direct hrcp intrinsic in some versions,
            // often handled by converting to float or using specific bf16 intrinsics
            return __float2bfloat16(1.0f / __bfloat162float(x));
        } else if constexpr (std::is_same_v<T, float>) {
            return __frcp_rd(x);
        } else {
            return static_cast<T>(1) / x;
        }
    }
} ReciprocalOp;
} // namespace op::reciprocal::cuda

#endif // __RECIPROCAL_CUDA_H__
