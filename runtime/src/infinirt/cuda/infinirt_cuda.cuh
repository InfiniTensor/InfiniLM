#ifndef __INFINIRT_CUDA_H__
#define __INFINIRT_CUDA_H__
#include "../infinirt_impl.h"

// NVIDIA namespace
namespace infinirt::cuda {
#ifdef ENABLE_NVIDIA_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::cuda

// ILUVATAR namespace
namespace infinirt::iluvatar {
#ifdef ENABLE_ILUVATAR_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::iluvatar

// QY namespace
namespace infinirt::qy {
#ifdef ENABLE_QY_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::qy

// HYGON namespace
namespace infinirt::hygon {
#ifdef ENABLE_HYGON_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::hygon

// ALI namespace
namespace infinirt::ali {
#ifdef ENABLE_ALI_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::ali

#endif // __INFINIRT_CUDA_H__
