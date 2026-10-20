#include <cstdlib>
#include <memory>

#include "base/all_gather.h"
#include "base/recv.h"
#include "base/send.h"
#include "devices/cpu/device_.h"

namespace infini::ccl {

struct UnsupportedOperation {};

template <>
struct BackendEnabled<Send, BackendType::kOmpi> : std::true_type {};

template <>
struct BackendEnabled<Recv, BackendType::kOmpi> : std::true_type {};

template <>
struct BackendEnabled<AllGather, BackendType::kOmpi> : std::true_type {};

template <>
struct AllGatherImpl<BackendType::kNccl, Device::Type::kNvidia> {
  static ReturnStatus Apply(const void *, void *, size_t, DataType,
                            Communicator *, void *) {
    return ReturnStatus::kInternalError;
  }
};

template <>
struct SendImpl<BackendType::kNccl, Device::Type::kNvidia> {
  static ReturnStatus Apply(const void *, size_t, DataType, int, Communicator *,
                            void *) {
    return ReturnStatus::kInternalError;
  }
};

template <>
struct RecvImpl<BackendType::kNccl, Device::Type::kNvidia> {
  static ReturnStatus Apply(void *, size_t, DataType, int, Communicator *,
                            void *) {
    return ReturnStatus::kInternalError;
  }
};

template <>
struct SendImpl<BackendType::kOmpi, Device::Type::kCpu> {
  static ReturnStatus Apply(const void *, size_t, DataType, int, Communicator *,
                            void *) {
    return ReturnStatus::kSuccess;
  }
};

template <>
struct RecvImpl<BackendType::kOmpi, Device::Type::kCpu> {
  static ReturnStatus Apply(void *, size_t, DataType, int, Communicator *,
                            void *) {
    return ReturnStatus::kSuccess;
  }
};

template <>
struct AllGatherImpl<BackendType::kOmpi, Device::Type::kCpu> {
  static ReturnStatus Apply(const void *, void *, size_t, DataType,
                            Communicator *, void *) {
    return ReturnStatus::kSuccess;
  }
};

std::unique_ptr<BackendCommInstance> MakeBackend(BackendType backend) {
  auto instance = std::make_unique<BackendCommInstance>();
  instance->type = backend;
  return instance;
}

bool TestUnsupportedCommunicator() {
  Communicator comm(Device::Type::kCpu, 0);
  comm.set_world_info(0, 2);
  comm.set_intra_comm(MakeBackend(BackendType::kMccl));

  float buffer = 0.0f;
  return Send::Execute<BackendType::kNccl, Device::Type::kNvidia>(
             &buffer, 1, DataType::kFloat32, 1, &comm, nullptr) ==
             ReturnStatus::kNotSupported &&
         Recv::Execute<BackendType::kNccl, Device::Type::kNvidia>(
             &buffer, 1, DataType::kFloat32, 1, &comm, nullptr) ==
             ReturnStatus::kNotSupported &&
         AllGather::Execute<BackendType::kNccl, Device::Type::kNvidia>(
             &buffer, &buffer, 1, DataType::kFloat32, &comm, nullptr) ==
             ReturnStatus::kNotSupported;
}

bool TestNoActiveBackend() {
  return Operation<UnsupportedOperation>::FindSupportedBackend(
             Device::Type::kCpu, {BackendType::kMccl}) == BackendType::kCount;
}

bool TestInterBackendFallback() {
  Communicator comm(Device::Type::kCpu, 0);
  comm.set_world_info(0, 2);
  comm.set_intra_comm(MakeBackend(BackendType::kMccl));
  comm.set_inter_comm(MakeBackend(BackendType::kOmpi));

  float buffer = 0.0f;
  return Send::Execute<BackendType::kNccl, Device::Type::kNvidia>(
             &buffer, 1, DataType::kFloat32, 1, &comm, nullptr) ==
             ReturnStatus::kSuccess &&
         Recv::Execute<BackendType::kNccl, Device::Type::kNvidia>(
             &buffer, 1, DataType::kFloat32, 1, &comm, nullptr) ==
             ReturnStatus::kSuccess &&
         AllGather::Execute<BackendType::kNccl, Device::Type::kNvidia>(
             &buffer, &buffer, 1, DataType::kFloat32, &comm, nullptr) ==
             ReturnStatus::kSuccess;
}

bool TestAllGatherNoBackend() {
  Communicator comm(Device::Type::kCpu, 0);
  comm.set_world_info(0, 2);

  float buffer = 0.0f;
  return AllGather::Execute<BackendType::kNccl, Device::Type::kNvidia>(
             &buffer, &buffer, 1, DataType::kFloat32, &comm, nullptr) ==
             ReturnStatus::kInternalError &&
         AllGather::Execute<BackendType::kNccl, Device::Type::kNvidia>(
             nullptr, nullptr, 0, DataType::kFloat32, &comm, nullptr) ==
             ReturnStatus::kSuccess;
}

}  // namespace infini::ccl

int main() {
  if (!infini::ccl::TestNoActiveBackend() ||
      !infini::ccl::TestUnsupportedCommunicator() ||
      !infini::ccl::TestInterBackendFallback() ||
      !infini::ccl::TestAllGatherNoBackend()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
