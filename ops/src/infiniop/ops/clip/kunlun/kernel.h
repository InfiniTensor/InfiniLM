#ifndef __CLIP_KUNLUN_KERNEL_H__
#define __CLIP_KUNLUN_KERNEL_H__
#include <xpu/kernel/xtdk_io.h>

namespace op::clip::kunlun {

typedef struct ClipOp {
public:
    static constexpr int num_inputs = 3;
    template <typename T>
    inline __device__ T operator()(const T *inputs) const {
        T x = inputs[0];
        T min_val = inputs[1];
        T max_val = inputs[2];
        return fmax(fmin(x, max_val), min_val);
    }

    // bfloat16 特化版本（使用 float 计算精度）
    inline __device__ bfloat16_t operator()(const bfloat16_t *inputs) const {
        float x_f = __bfloat162float(inputs[0]);
        float min_val_f = __bfloat162float(inputs[1]);
        float max_val_f = __bfloat162float(inputs[2]);
        float result_f = fmax(fmin(x_f, max_val_f), min_val_f);
        return __float2bfloat16(result_f);
    }
} ClipOp;

} // namespace op::clip::kunlun

#endif // __CLIP_KUNLUN_KERNEL_H__
