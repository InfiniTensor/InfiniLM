#ifndef __CLIP_CUDA_API_H__
#define __CLIP_CUDA_API_H__

#include "../../../elementwise/cuda/elementwise_cuda_api.cuh"
#include "../clip.h"

CLIP_DESCRIPTOR(clip, cuda)

namespace op::clip::cuda {

infiniStatus_t createClipDescriptor(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec);

} // namespace op::clip::cuda

#endif // __CLIP_CUDA_API_H__
