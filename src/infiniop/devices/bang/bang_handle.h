#ifndef BANG_HANDLE_H
#define BANG_HANDLE_H

#include "infiniop/handle.h"

struct InfiniopBangHandle;
typedef struct InfiniopBangHandle *infiniopBangHandle_t;

infiniopStatus_t createBangHandle(infiniopBangHandle_t *handle_ptr,
                                  int device_id);
infiniopStatus_t destroyBangHandle(infiniopBangHandle_t handle);

#endif
