#ifndef INFINICCL_ASCEND_H_
#define INFINICCL_ASCEND_H_

#include "../infiniccl_impl.h"

#if defined(ENABLE_ASCEND_API) && defined(ENABLE_CCL)
INFINICCL_DEVICE_API_IMPL(ascend)
#else
INFINICCL_DEVICE_API_NOOP(ascend)
#endif

#endif /* INFINICCL_ASCEND_H_ */
