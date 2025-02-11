#ifndef ASCEND_HANDLE_H
#define ASCEND_HANDLE_H

#include "common_ascend.h"
#include "device.h"
#include "status.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>
#include <memory>

struct AscendContext {
    Device device;
    int device_id;
};
typedef struct AscendContext *AscendHandle_t;

infiniopStatus_t createAscendHandle(AscendHandle_t *handle_ptr, int device_id);

infiniopStatus_t deleteAscendHandle(AscendHandle_t handle_ptr);

#endif
