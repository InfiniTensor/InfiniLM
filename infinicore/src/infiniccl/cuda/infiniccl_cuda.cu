#include "infiniccl_cuda.h"

#include <cuda_runtime.h>
#include <iostream>
#include <nccl.h>
#include <vector>

#include "../../utils.h"

#define CHECK_NCCL(API__) CHECK_INTERNAL(API__, ncclSuccess)

inline cudaStream_t getCudaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<cudaStream_t>(stream);
}

inline ncclDataType_t getNcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return ncclFloat;
    case INFINI_DTYPE_F16:
        return ncclHalf;
    default:
        std::abort();
        return ncclHalf;
    }
}

inline ncclRedOp_t getNcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return ncclSum;
    case INFINICCL_PROD:
        return ncclProd;
    case INFINICCL_MAX:
        return ncclMax;
    case INFINICCL_MIN:
        return ncclMin;
    case INFINICCL_AVG:
        return ncclAvg;
    default:
        std::abort();
        return ncclSum;
    }
}

inline ncclComm_t getNcclComm(infinicclComm_t comm) {
    return static_cast<ncclComm_t>(comm->comm);
}

namespace infiniccl::cuda {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<ncclComm_t> nccl_comms(ndevice);
    CHECK_NCCL(ncclCommInitAll(nccl_comms.data(), ndevice, (int const *)device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_NVIDIA, device_ids[i], (void *)(nccl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_NCCL(ncclCommDestroy(getNcclComm(comm)));
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

    CHECK_NCCL(ncclAllReduce(sendbuf, recvbuf, count, getNcclDtype(datatype),
                             getNcclRedOp(op), getNcclComm(comm), getCudaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::cuda
