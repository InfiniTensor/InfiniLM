#ifndef __INFINICCL_API_H__
#define __INFINICCL_API_H__

#include "infinirt.h"

typedef enum {
    INFINICCL_SUM = 0,
    INFINICCL_PROD = 1,
    INFINICCL_MAX = 2,
    INFINICCL_MIN = 3,
    INFINICCL_AVG = 4,
} infinicclReduceOp_t;

struct InfinicclComm;

typedef struct InfinicclComm *infinicclComm_t;

__C __export infiniStatus_t infinicclCommInitAll(
    infiniDevice_t device_type,
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids);

__C __export infiniStatus_t infinicclCommDestroy(infinicclComm_t comm);

__C __export infiniStatus_t infinicclAllReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t dataype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream);

#endif
