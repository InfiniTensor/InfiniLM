#ifndef INFINICCL_CAMBRICON_H_
#define INFINICCL_CAMBRICON_H_

#include "../infiniccl_impl.h"

#if defined(ENABLE_CAMBRICON_API) && defined(ENABLE_CCL)
INFINICCL_DEVICE_API_IMPL(cambricon)
#else
INFINICCL_DEVICE_API_NOOP(cambricon)
#endif

#endif /* INFINICCL_CAMBRICON_H_ */
