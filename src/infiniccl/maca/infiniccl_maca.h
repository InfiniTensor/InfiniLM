#ifndef INFINICCL_MACA_H_
#define INFINICCL_MACA_H_

#include "../infiniccl_impl.h"

#if defined(ENABLE_METAX_API) && defined(ENABLE_CCL)
INFINICCL_DEVICE_API_IMPL(maca)
#else
INFINICCL_DEVICE_API_NOOP(maca)
#endif

#endif /* INFINICCL_MACA_H_ */
