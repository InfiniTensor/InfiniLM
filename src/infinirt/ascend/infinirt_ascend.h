#ifndef __INFINIRT_ASCEND_H__
#define __INFINIRT_ASCEND_H__
#include "../infinirt_impl.h"

namespace infinirt::ascend {
#ifdef ENABLE_ASCEND_API
infiniStatus_t init();
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::ascend

#endif // __INFINIRT_ASCEND_H__
