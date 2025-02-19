#ifndef __INFINIOP_KUNLUN_HANDLE_H__
#define __INFINIOP_KUNLUN_HANDLE_H__

#include "infiniop/handle.h"

struct InfiniopKunlunHandle;

typedef struct InfiniopKunlunHandle *infiniopKunlunHandle_t;

infiniopStatus_t createKunlunHandle(infiniopKunlunHandle_t *handle_ptr);
infiniopStatus_t destroyKunlunHandle(infiniopKunlunHandle_t handle);

#endif
