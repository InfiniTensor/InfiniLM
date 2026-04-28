#ifndef __CLIP_CPU_H__
#define __CLIP_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "infiniop/ops/clip.h"

ELEMENTWISE_DESCRIPTOR(clip, cpu)

namespace op::clip::cpu {

typedef struct ClipOp {
public:
    static constexpr size_t num_inputs = 3;

    template <typename T>
    T operator()(const T &x, const T &min_val, const T &max_val) const {
        return std::max(std::min(x, max_val), min_val);
    }
} ClipOp;

} // namespace op::clip::cpu

#endif // __CLIP_CPU_H__
