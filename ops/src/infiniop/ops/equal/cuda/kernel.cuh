#ifndef __EQUAL_CUDA_H__
#define __EQUAL_CUDA_H__

#include <type_traits>

namespace op::equal::cuda {

typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename Tout, typename Tin0, typename Tin1>
    __device__ __forceinline__ bool operator()(const Tin0 &a, const Tin1 &b) const {
        if constexpr (std::is_same_v<Tin0, Tin1>) {
            if constexpr (std::is_same_v<Tin0, half2>) {
                static_assert(!std::is_same_v<Tin0, half2>, "half2 is not supported for mixed output dtype");
            } else if constexpr (std::is_same_v<Tin0, half>) {
                return static_cast<Tout>(__heq(a, b));
            } else {
                return static_cast<Tout>(a == b);
            }
        } else {
            return false;
        }
    }
} EqualOp;

} // namespace op::equal::cuda

#endif
