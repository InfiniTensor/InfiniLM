#ifndef __CLIP_CPU_H__
#define __CLIP_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "../clip.h"

CLIP_DESCRIPTOR(clip, cpu)

namespace op::clip::cpu {

typedef struct ClipOp {
public:
    static constexpr size_t num_inputs = 3;

    template <typename T>
    T operator()(const T &x, const T &min_val, const T &max_val) const {
        return std::max(std::min(x, max_val), min_val);
    }
} ClipOp;

// Create clip descriptor
infiniStatus_t createClipDescriptor(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec);

} // namespace op::clip::cpu

#endif // __CLIP_CPU_H__
