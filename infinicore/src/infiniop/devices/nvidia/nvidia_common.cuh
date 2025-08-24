#ifndef __INFINIOP_CUDA_COMMON_CUH__
#define __INFINIOP_CUDA_COMMON_CUH__

#include "infinicore.h"
#include "nvidia_handle.cuh"

namespace device::nvidia {

#ifdef ENABLE_CUDNN_API
cudnnDataType_t getCudnnDtype(infiniDtype_t dt);
#endif

} // namespace device::nvidia

#endif // __INFINIOP_CUDA_COMMON_CUH__
