#ifndef __EQUAL_MOORE_KERNEL_H__
#define __EQUAL_MOORE_KERNEL_H__

#include <type_traits>

namespace op::equal::moore {

typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename Tout, typename Tin0, typename Tin1>
    __device__ __forceinline__ bool operator()(const Tin0 &a, const Tin1 &b) const {
        if constexpr (std::is_same_v<Tin0, Tin1>) {
            if constexpr (std::is_same_v<Tin0, half>) {
                return __half2float(a) == __half2float(b);
            } else if constexpr (std::is_same_v<Tin0, cuda_bfloat16>) {
                return __bfloat162float(a) == __bfloat162float(b);
            } else {
                return a == b;
            }
        } else {
            return false;
        }
    }
} EqualOp;

} // namespace op::equal::moore

#endif // __EQUAL_MOORE_KERNEL_H__
