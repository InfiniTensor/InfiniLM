#ifndef __CLIP_CUDA_H__
#define __CLIP_CUDA_H__

namespace op::clip::cuda {

typedef struct ClipOp {
public:
    static constexpr size_t num_inputs = 3;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const T &min_val, const T &max_val) const {
        if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, cuda_bfloat162>) {
#ifndef ENABLE_ILUVATAR_API
            return __hmax2(__hmin2(x, max_val), min_val);
#else
            return {std::clamp(x.x, min_val.x, max_val.x), std::clamp(x.y, min_val.y, max_val.y)};
#endif
        } else {
            return std::clamp(x, min_val, max_val);
        }
    }
} ClipOp;
} // namespace op::clip::cuda

#endif // __CLIP_CUDA_H__
