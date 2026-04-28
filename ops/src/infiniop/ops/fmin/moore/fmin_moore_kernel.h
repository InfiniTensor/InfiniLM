#ifndef __FMIN_MOORE_KERNEL_H__
#define __FMIN_MOORE_KERNEL_H__

namespace op::fmin::moore {
typedef struct FminOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hmin2(a, b);
        } else if constexpr (std::is_same_v<T, half>) {
            return __hmin(a, b);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float a_f = __bfloat162float(a);
            float b_f = __bfloat162float(b);
            return fminf(a_f, b_f);
        } else if constexpr (std::is_same_v<T, float>) {
            return fminf(a, b);
        } else {
            return a < b ? a : b;
        }
    }
} FminOp;
} // namespace op::fmin::moore

#endif // __FMIN_MOORE_KERNEL_H__
