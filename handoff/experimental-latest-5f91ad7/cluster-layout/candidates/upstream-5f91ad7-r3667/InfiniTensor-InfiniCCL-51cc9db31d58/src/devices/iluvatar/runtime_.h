#ifndef INFINI_CCL_DEVICES_ILUVATAR_RUNTIME_H_
#define INFINI_CCL_DEVICES_ILUVATAR_RUNTIME_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "devices/cuda/runtime_.h"
#include "devices/iluvatar/device_.h"
#include "logging.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <>
struct Runtime<Device::Type::kIluvatar>
    : CudaRuntime<Runtime<Device::Type::kIluvatar>> {
  using Stream = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kIluvatar;

  static constexpr auto Check =
      [](auto status, ReturnStatus err_code = ReturnStatus::kSystemError) {
        if (status != cudaSuccess) {
          LOG(cudaGetErrorString(static_cast<cudaError_t>(status)));
          return err_code;
        }
        return ReturnStatus::kSuccess;
      };

  static constexpr auto Malloc = [](auto &&...args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = cudaMemcpy;

  static constexpr auto Free = cudaFree;

  static constexpr auto MemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto Memset = cudaMemset;

  static constexpr auto GetDevice = cudaGetDevice;

  static constexpr auto SetDevice = cudaSetDevice;

  static constexpr auto DeviceSynchronize = cudaDeviceSynchronize;

  static constexpr auto StreamSynchronize = cudaStreamSynchronize;
};

static_assert(Runtime<Device::Type::kIluvatar>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_ILUVATAR_RUNTIME_H_
