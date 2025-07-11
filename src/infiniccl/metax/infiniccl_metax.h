#ifndef INFINICCL_METAX_H_
#define INFINICCL_METAX_H_

#include "../infiniccl_impl.h"

#if defined(ENABLE_METAX_API) && defined(ENABLE_CCL)
INFINICCL_DEVICE_API_IMPL(metax)
#else
INFINICCL_DEVICE_API_NOOP(metax)
#endif

#endif /* INFINICCL_METAX_H_ */
