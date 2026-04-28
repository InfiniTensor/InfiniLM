#include "infiniccl.h"

#include "./ascend/infiniccl_ascend.h"
#include "./cambricon/infiniccl_cambricon.h"
#include "./cuda/infiniccl_cuda.h"
#include "./kunlun/infiniccl_kunlun.h"
#include "./metax/infiniccl_metax.h"
#include "./moore/infiniccl_moore.h"

__INFINI_C infiniStatus_t infinicclCommInitAll(
    infiniDevice_t device_type,
    infinicclComm_t *comms,
    int ndevice,
    const int *device_ids) {

#define COMM_INIT_ALL(CASE_, NAMESPACE_) \
    case CASE_:                          \
        return infiniccl::NAMESPACE_::commInitAll(comms, ndevice, device_ids)

    switch (device_type) {
        COMM_INIT_ALL(INFINI_DEVICE_NVIDIA, cuda);
        COMM_INIT_ALL(INFINI_DEVICE_ILUVATAR, cuda);
        COMM_INIT_ALL(INFINI_DEVICE_QY, cuda);
        COMM_INIT_ALL(INFINI_DEVICE_HYGON, cuda);
        COMM_INIT_ALL(INFINI_DEVICE_ASCEND, ascend);
        COMM_INIT_ALL(INFINI_DEVICE_CAMBRICON, cambricon);
        COMM_INIT_ALL(INFINI_DEVICE_METAX, metax);
        COMM_INIT_ALL(INFINI_DEVICE_MOORE, moore);
        COMM_INIT_ALL(INFINI_DEVICE_KUNLUN, kunlun);
        COMM_INIT_ALL(INFINI_DEVICE_ALI, cuda);
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef COMM_INIT_ALL
}

__INFINI_C infiniStatus_t infinicclCommDestroy(infinicclComm_t comm) {
    if (comm == nullptr) {
        return INFINI_STATUS_SUCCESS;
    }

#define COMM_DESTROY(CASE_, NAMESPACE_) \
    case CASE_:                         \
        return infiniccl::NAMESPACE_::commDestroy(comm)

    switch (comm->device_type) {
        COMM_DESTROY(INFINI_DEVICE_NVIDIA, cuda);
        COMM_DESTROY(INFINI_DEVICE_ILUVATAR, cuda);
        COMM_DESTROY(INFINI_DEVICE_QY, cuda);
        COMM_DESTROY(INFINI_DEVICE_HYGON, cuda);
        COMM_DESTROY(INFINI_DEVICE_ASCEND, ascend);
        COMM_DESTROY(INFINI_DEVICE_CAMBRICON, cambricon);
        COMM_DESTROY(INFINI_DEVICE_METAX, metax);
        COMM_DESTROY(INFINI_DEVICE_MOORE, moore);
        COMM_DESTROY(INFINI_DEVICE_KUNLUN, kunlun);
        COMM_DESTROY(INFINI_DEVICE_ALI, cuda);
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef COMM_DESTROY
}

__INFINI_C infiniStatus_t infinicclAllReduce(
    void *sendbuf,
    void *recvbuf,
    size_t count,
    infiniDtype_t dataype,
    infinicclReduceOp_t op,
    infinicclComm_t comm,
    infinirtStream_t stream) {

    if (comm == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

#define ALL_REDUCE(CASE_, NAMESPACE_) \
    case CASE_:                       \
        return infiniccl::NAMESPACE_::allReduce(sendbuf, recvbuf, count, dataype, op, comm, stream)

    switch (comm->device_type) {
        ALL_REDUCE(INFINI_DEVICE_NVIDIA, cuda);
        ALL_REDUCE(INFINI_DEVICE_ILUVATAR, cuda);
        ALL_REDUCE(INFINI_DEVICE_QY, cuda);
        ALL_REDUCE(INFINI_DEVICE_HYGON, cuda);
        ALL_REDUCE(INFINI_DEVICE_ASCEND, ascend);
        ALL_REDUCE(INFINI_DEVICE_CAMBRICON, cambricon);
        ALL_REDUCE(INFINI_DEVICE_METAX, metax);
        ALL_REDUCE(INFINI_DEVICE_MOORE, moore);
        ALL_REDUCE(INFINI_DEVICE_KUNLUN, kunlun);
        ALL_REDUCE(INFINI_DEVICE_ALI, cuda);

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef ALL_REDUCE
}
