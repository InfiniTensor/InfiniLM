#ifndef __INFINIOP_ASCEND_HANDLE_H__
#define __INFINIOP_ASCEND_HANDLE_H__

#include "../../handle.h"
#include "infinicore.h"

struct InfiniopAscendHandle;
typedef struct InfiniopAscendHandle *infiniopAscendHandle_t;

infiniStatus_t createAscendHandle(infiniopAscendHandle_t *handle_ptr);

infiniStatus_t destroyAscendHandle(infiniopAscendHandle_t handle_ptr);

#endif
