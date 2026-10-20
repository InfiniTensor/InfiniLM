#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_NVIDIA_API_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_NVIDIA_API_H_

#include "backends/ccl/nccl/api.h"
#include "backends/ccl/common/comm_instance.h"
#include "base/all_reduce.h"
#include "devices/nvidia/runtime_.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kNccl, Device::Type::kNvidia>
    : NcclApi<Device::Type::kNvidia> {};

template <>
struct AllReduceCaptureImpl<BackendType::kNccl, Device::Type::kNvidia> {
  static ReturnStatus Query(Communicator *comm, int *supported) {
    using Api = CclApi<BackendType::kNccl, Device::Type::kNvidia>;
    using DeviceRuntime = Runtime<Device::Type::kNvidia>;
    using CommInstance = CclCommInstance<Api>;

    if (comm->device_type() != Device::Type::kNvidia ||
        comm->device_id() < 0 ||
        comm->intra_comm_backend() != BackendType::kNccl) {
      return ReturnStatus::kInvalidArgument;
    }
    auto *intra = static_cast<CommInstance *>(comm->intra_comm());
    if (!intra || !intra->handle) {
      return ReturnStatus::kInternalError;
    }
    int current_device = -1;
    auto status = DeviceRuntime::Check(DeviceRuntime::GetDevice(&current_device));
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
    if (current_device != comm->device_id()) {
      return ReturnStatus::kInvalidArgument;
    }

    if (NCCL_VERSION_CODE < 20900 || CUDART_VERSION < 11030) {
      return ReturnStatus::kSuccess;
    }
    int nccl_version = 0;
    status = Api::Check(ncclGetVersion(&nccl_version));
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
    int cuda_version = 0;
    status = DeviceRuntime::Check(cudaRuntimeGetVersion(&cuda_version));
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
    int driver_version = 0;
    status = DeviceRuntime::Check(cudaDriverGetVersion(&driver_version));
    if (status != ReturnStatus::kSuccess) {
      return status;
    }
    if (nccl_version >= 20900 && cuda_version >= 11030 &&
        driver_version >= 11030) {
      *supported = 1;
    }
    return ReturnStatus::kSuccess;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_NVIDIA_API_H_
