#ifndef __ADD_CUDA_H__
#define __ADD_CUDA_H__

namespace op::add::cuda {
typedef struct AddOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            return __hadd2(a, b);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return __hadd(a, b);
        } else if constexpr (std::is_same_v<T, float>) {
            return __fadd_rd(a, b);
        } else {
            return a + b;
        }
    }
} AddOp;
} // namespace op::add::cuda

#endif // __ADD_CUDA_H__
