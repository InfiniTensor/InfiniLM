#ifndef INFINICCL_MOORE_H_
#define INFINICCL_MOORE_H_

#include "../infiniccl_impl.h"

#if defined(ENABLE_MOORE_API) && defined(ENABLE_CCL)
INFINICCL_DEVICE_API_IMPL(moore)
#else
INFINICCL_DEVICE_API_NOOP(moore)
#endif

#endif /* INFINICCL_MOORE_H_ */
