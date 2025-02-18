#ifndef __RANDOM_SAMPLE_ASCEND_H__
#define __RANDOM_SAMPLE_ASCEND_H__

#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../utils.h"
#include "random_sample_ascend_api.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnnop/aclnn_topk.h>

struct InfiniopRandomSampleAscendDescriptor {
    infiniDevice_t device;
    int device_id;
    aclnnTensorDescriptor_t pDesc, topkValDesc, topkIdxDesc, resDesc;

    InfiniopRandomSampleAscendDescriptor(infiniDevice_t device_);
};

extern "C" void
random_sample_do(void *p, void *res, void *topkAddr, void *topkIdxAddr,
                 int32_t topk, int32_t voc, float topp, float temper,
                 float random, int dtype, void *stream);

#endif
