#ifndef INFINICCL_KUNLUN_H_
#define INFINICCL_KUNLUN_H_

#include "../infiniccl_impl.h"

#if defined(ENABLE_KUNLUN_API) && defined(ENABLE_CCL)
INFINICCL_DEVICE_API_IMPL(kunlun)
#else
INFINICCL_DEVICE_API_NOOP(kunlun)
#endif

#endif /* INFINICCL_KUNLUN_H_ */
