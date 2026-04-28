#ifndef __INFINIRT_MUSA_H__
#define __INFINIRT_MUSA_H__
#include "../infinirt_impl.h"

namespace infinirt::musa {
#ifdef ENABLE_MOORE_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::musa

#endif // __INFINIRT_MUSA_H__
