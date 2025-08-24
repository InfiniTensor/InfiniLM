#ifndef __INFINIRT_CPU_H__
#define __INFINIRT_CPU_H__
#include "../infinirt_impl.h"

namespace infinirt::cpu {
#ifdef ENABLE_CPU_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::cpu

#endif // __INFINIRT_CPU_H__
