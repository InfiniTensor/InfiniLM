#ifndef __INFINIRT_CUDA_H__
#define __INFINIRT_CUDA_H__
#include "../infinirt_impl.h"

namespace infinirt::cuda {
#ifdef ENABLE_CUDA_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::cuda

#endif // __INFINIRT_CUDA_H__
