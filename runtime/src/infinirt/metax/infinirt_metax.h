#ifndef __INFINIRT_MACA_H__
#define __INFINIRT_MACA_H__
#ifdef ENABLE_METAX_API
#include "../../infiniop/devices/metax/metax_ht2mc.h"
#endif
#include "../infinirt_impl.h"

namespace infinirt::metax {
#ifdef ENABLE_METAX_API
INFINIRT_DEVICE_API_IMPL
#else
INFINIRT_DEVICE_API_NOOP
#endif
} // namespace infinirt::metax

#endif // __INFINIRT_MACA_H__
