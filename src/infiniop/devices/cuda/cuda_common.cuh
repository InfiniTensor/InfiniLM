#ifndef __INFINIOP_CUDA_COMMON_CUH__
#define __INFINIOP_CUDA_COMMON_CUH__

#include "cuda_handle.cuh"
#include "infinicore.h"

namespace device::cuda {

cudnnDataType_t getCudnnDtype(infiniDtype_t dt);

} // namespace device::cuda

#endif // __INFINIOP_CUDA_COMMON_CUH__
