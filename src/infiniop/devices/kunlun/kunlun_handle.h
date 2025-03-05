#ifndef __INFINIOP_KUNLUN_HANDLE_H__
#define __INFINIOP_KUNLUN_HANDLE_H__

#include "../../handle.h"

struct InfiniopKunlunHandle;

typedef struct InfiniopKunlunHandle *infiniopKunlunHandle_t;

infiniStatus_t createKunlunHandle(infiniopKunlunHandle_t *handle_ptr);
infiniStatus_t destroyKunlunHandle(infiniopKunlunHandle_t handle);

#endif // __INFINIOP_KUNLUN_HANDLE_H__
