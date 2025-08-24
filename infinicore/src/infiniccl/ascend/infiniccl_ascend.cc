#include "infiniccl_ascend.h"

#include "../../utils.h"

#include <acl/acl.h>
#include <hccl.h>

#include <iostream>
#include <vector>

#define CHECK_HCCL(API__) CHECK_INTERNAL(API__, HCCL_SUCCESS)

inline aclrtStream getAscendStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<aclrtStream>(stream);
}

inline HcclComm getHcclComm(infinicclComm_t comm) {
    return static_cast<HcclComm>(comm->comm);
}

inline HcclDataType getAscendDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return HCCL_DATA_TYPE_FP32;
    case INFINI_DTYPE_F16:
        return HCCL_DATA_TYPE_FP16;
    default:
        std::cerr << "Unsupported data type: " << datatype << std::endl;
        std::abort();
        return HCCL_DATA_TYPE_FP16;
    }
}

inline HcclReduceOp getHcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return HCCL_REDUCE_SUM;
    case INFINICCL_PROD:
        return HCCL_REDUCE_PROD;
    case INFINICCL_MAX:
        return HCCL_REDUCE_MAX;
    case INFINICCL_MIN:
        return HCCL_REDUCE_MIN;
    default:
        std::abort();
        return HCCL_REDUCE_SUM;
    }
}

namespace infiniccl::ascend {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {
    // Ascend requires all devices to be initialized before calling HcclCommInitAll.
    for (int i = ndevice - 1; i >= 0; i--) {
        aclrtSetDevice(device_ids[i]);
    }

    std::vector<HcclComm> hccl_comms(ndevice);
    CHECK_HCCL(HcclCommInitAll(ndevice, (int32_t *)device_ids, hccl_comms.data()));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_ASCEND, device_ids[i], (void *)(hccl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_HCCL(HcclCommDestroy(getHcclComm(comm)));
    delete comm;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t allReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t datatype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (datatype != INFINI_DTYPE_F32 && datatype != INFINI_DTYPE_F16) {
        return INFINI_STATUS_BAD_PARAM;
    }

    CHECK_HCCL(HcclAllReduce(sendbuf, recvbuf, (uint64_t)count,
                             getAscendDtype(datatype), getHcclRedOp(op),
                             getHcclComm(comm), getAscendStream(stream)));

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::ascend
