#include "infiniccl_moore.h"

#include "../../utils.h"

#include <mccl.h>
#include <musa_runtime.h>

#include <iostream>
#include <vector>

#define CHECK_MCCL(API__) CHECK_INTERNAL(API__, mcclSuccess)

inline musaStream_t getMusaStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return static_cast<musaStream_t>(stream);
}

inline mcclDataType_t getMcclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return mcclFloat;
    case INFINI_DTYPE_F16:
        return mcclHalf;

#if MARCH_TYPE == 310
    case INFINI_DTYPE_BF16:
        return mcclBfloat16;
#endif

    default:
        std::abort();
        return mcclHalf;
    }
}

inline mcclRedOp_t getMcclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return mcclSum;
    case INFINICCL_PROD:
        return mcclProd;
    case INFINICCL_MAX:
        return mcclMax;
    case INFINICCL_MIN:
        return mcclMin;
    case INFINICCL_AVG:
        return mcclAvg;
    default:
        std::abort();
        return mcclSum;
    }
}

inline mcclComm_t getMcclComm(infinicclComm_t comm) {
    return static_cast<mcclComm_t>(comm->comm);
}

namespace infiniccl::moore {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<mcclComm_t> mccl_comms(ndevice);
    CHECK_MCCL(mcclCommInitAll(mccl_comms.data(), ndevice, (int const *)device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_MOORE, device_ids[i], (void *)(mccl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_MCCL(mcclCommDestroy(getMcclComm(comm)));
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

#if MARCH_TYPE == 310
    CHECK_DTYPE(datatype,
                INFINI_DTYPE_F32,
                INFINI_DTYPE_F16,
                INFINI_DTYPE_BF16);
#else
    CHECK_DTYPE(datatype,
                INFINI_DTYPE_F32,
                INFINI_DTYPE_F16);
#endif

    CHECK_MCCL(mcclAllReduce(sendbuf, recvbuf, count, getMcclDtype(datatype),
                             getMcclRedOp(op), getMcclComm(comm), getMusaStream(stream)));

    return INFINI_STATUS_SUCCESS;
}
} // namespace infiniccl::moore
