#include "infiniccl_maca.h"

#include "../../utils.h"

#include <hccl.h>
#include <hcr/hc_runtime_api.h>

#include <iostream>
#include <vector>

#define CHECK_HCCL(API__) CHECK_INTERNAL(API__, hcclSuccess)

inline hcStream_t getMacaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<hcStream_t>(stream);
}

inline hcclDataType_t getHcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return hcclFloat;
    case INFINI_DTYPE_F16:
        return hcclHalf;
    default:
        std::abort();
        return hcclHalf;
    }
}

inline hcclRedOp_t getHcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return hcclSum;
    case INFINICCL_PROD:
        return hcclProd;
    case INFINICCL_MAX:
        return hcclMax;
    case INFINICCL_MIN:
        return hcclMin;
    case INFINICCL_AVG:
        return hcclAvg;
    default:
        std::abort();
        return hcclSum;
    }
}

inline hcclComm_t getHcclComm(infinicclComm_t comm) {
    return static_cast<hcclComm_t>(comm->comm);
}

namespace infiniccl::maca {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<hcclComm_t> hccl_comms(ndevice);
    CHECK_HCCL(hcclCommInitAll(hccl_comms.data(), ndevice, (int const *)device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_METAX, device_ids[i], (void *)(hccl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_HCCL(hcclCommDestroy(getHcclComm(comm)));
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

    CHECK_HCCL(hcclAllReduce(sendbuf, recvbuf, count, getHcclDtype(datatype),
                             getHcclRedOp(op), getHcclComm(comm), getMacaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::maca
