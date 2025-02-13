#ifndef __INFINIOP_ASCEND_HANDLE_H__
#define __INFINIOP_ASCEND_HANDLE_H__

#include "infinicore.h"
#include "infiniop/handle.h"

struct InfiniopAscendHandle;
typedef struct InfiniopAscendHandle *infiniopAscendHandle_t;

infiniopStatus_t createAscendHandle(infiniopAscendHandle_t *handle_ptr,
                                    int device_id);

infiniopStatus_t destroyAscendHandle(infiniopAscendHandle_t handle_ptr);

#endif
