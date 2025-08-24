#ifndef __INFINIRT_KUNLUN_H__
#define __INFINIRT_KUNLUN_H__
#include "../infinirt_impl.h"

namespace infinirt::kunlun {
#ifdef ENABLE_KUNLUN_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::kunlun

#endif // __INFINIRT_KUNLUN_H__
