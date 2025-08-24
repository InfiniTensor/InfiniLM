#ifndef __INFINIRT_BANG_H__
#define __INFINIRT_BANG_H__
#include "../infinirt_impl.h"

namespace infinirt::bang {
#ifdef ENABLE_CAMBRICON_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::bang

#endif // __INFINIRT_BANG_H__
