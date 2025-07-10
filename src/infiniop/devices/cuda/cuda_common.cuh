#ifndef __INFINIOP_CUDA_COMMON_CUH__
#define __INFINIOP_CUDA_COMMON_CUH__

#include "cuda_handle.cuh"
#include "infinicore.h"

namespace device::cuda {

#ifdef ENABLE_CUDNN_API
cudnnDataType_t getCudnnDtype(infiniDtype_t dt);
#endif

} // namespace device::cuda

#endif // __INFINIOP_CUDA_COMMON_CUH__
