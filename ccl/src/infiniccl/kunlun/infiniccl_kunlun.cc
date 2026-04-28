#include "infiniccl_kunlun.h"

#include "../../utils.h"

#include <bkcl.h>

#include <iostream>
#include <vector>

#define CHECK_BKCL(API__) CHECK_INTERNAL(API__, BKCL_SUCCESS)

typedef XPUStream kunlunStream_t;
typedef BKCLContext_t bkclComm_t;

inline kunlunStream_t getKunlunStream(infinirtStream_t stream) {
    if (stream == nullptr) {
        return 0;
    }
    return reinterpret_cast<kunlunStream_t>(stream);
}

inline bkclComm_t getBkclComm(infinicclComm_t comm) {
    return reinterpret_cast<bkclComm_t>(comm->comm);
}

inline BKCLDataType getBkclDtype(infiniDtype_t datatype) {
    switch (datatype) {
    case INFINI_DTYPE_F32:
        return BKCL_FLOAT;
    case INFINI_DTYPE_F16:
        return BKCL_FLOAT16;
    case INFINI_DTYPE_BF16:
        return BKCL_BFLOAT16;
    default:
        std::cerr << "Unsupported data type: " << datatype << std::endl;
        std::abort();
        return BKCL_FLOAT16;
    }
}

inline BKCLOp getBkclRedOp(infinicclReduceOp_t op) {
    switch (op) {
    case INFINICCL_SUM:
        return BKCL_ADD;
    case INFINICCL_PROD:
        return BKCL_PRODUCT;
    case INFINICCL_MAX:
        return BKCL_MAX;
    case INFINICCL_MIN:
        return BKCL_MIN;
    default:
        std::abort();
        return BKCL_ADD;
    }
}

namespace infiniccl::kunlun {

infiniStatus_t commInitAll(
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {
    std::vector<bkclComm_t> bkcl_comms(ndevice);
    CHECK_BKCL(bkcl_comm_init_all(bkcl_comms.data(), ndevice, device_ids));

    for (int i = 0; i < ndevice; i++) {
        comms[i] = new InfinicclComm{INFINI_DEVICE_KUNLUN, device_ids[i], (void *)(bkcl_comms[i])};
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t commDestroy(infinicclComm_t comm) {
    CHECK_BKCL(bkcl_destroy_context(getBkclComm(comm)));
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
    CHECK_BKCL(bkcl_all_reduce(
        getBkclComm(comm),
        sendbuf,
        recvbuf,
        count,
        getBkclDtype(datatype),
        getBkclRedOp(op),
        getKunlunStream(stream)));

    return INFINI_STATUS_SUCCESS;
}

} // namespace infiniccl::kunlun
