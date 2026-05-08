#include "infiniccl_cambricon.h"

#include "../../utils.h"
#include <cncl.h>
#include <cnrt.h>
#include <iostream>
#include <vector>

#define CHECK_CNCL(API__) CHECK_INTERNAL(API__, CNCL_RET_SUCCESS)

inline cnrtQueue_t getCambriconStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return (cnrtQueue_t)(0);
    }
    return static_cast<cnrtQueue_t>(stream);
}

inline cnclComm_t getCnclComm(infinicclComm_t comm) {
    return static_cast<cnclComm_t>(comm->comm);
}

inline cnclDataType_t getCnclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return cnclFloat32;
    case INFINI_DTYPE_F16:
        return cnclFloat16;
    case INFINI_DTYPE_BF16:
        return cnclBfloat16;
    default:
        std::cerr << "Unsupported data type: " << datatype << std::endl;
        std::abort();
        return cnclFloat16;
    }
}

inline cnclReduceOp_t getCnclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return cnclSum;
    case INFINICCL_PROD:
        return cnclProd;
    case INFINICCL_MAX:
        return cnclMax;
    case INFINICCL_MIN:
        return cnclMin;
    default:
        std::abort();
        return cnclSum;
    }
}

namespace infiniccl::cambricon {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

    std::vector<cnclComm_t> cncl_comms(ndevice);
    std::vector<int> rank_list(ndevice);

    for (int i = 0; i < ndevice; i++) {
        rank_list[i] = i;
        CHECK_INTERNAL(cnrtSetDevice(device_ids[i]), cnrtSuccess);
    }

    CHECK_CNCL(cnclInitComms(cncl_comms.data(), ndevice,
                             (int const *)device_ids, rank_list.data(),
                             ndevice, nullptr));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_CAMBRICON, device_ids[i], (void *)(cncl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_CNCL(cnclFreeComm(getCnclComm(comm)));
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

    CHECK_DTYPE(datatype, INFINI_DTYPE_F32, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);

    CHECK_CNCL(cnclAllReduce(sendbuf, recvbuf, count, getCnclDtype(datatype),
                             getCnclRedOp(op), getCnclComm(comm),
                             getCambriconStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

} // namespace infiniccl::cambricon
