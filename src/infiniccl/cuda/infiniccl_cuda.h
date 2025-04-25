#ifndef INFINICCL_CUDA_H_
#define INFINICCL_CUDA_H_

#include "../infiniccl_impl.h"

// Windows does not support CUDA
#if defined(ENABLE_CUDA_API) && defined(ENABLE_CCL) && !defined(_WIN32)
INFINICCL_DEVICE_API_IMPL(cuda)
#else
INFINICCL_DEVICE_API_NOOP(cuda)
#endif

#endif /* INFINICCL_CUDA_H_ */
