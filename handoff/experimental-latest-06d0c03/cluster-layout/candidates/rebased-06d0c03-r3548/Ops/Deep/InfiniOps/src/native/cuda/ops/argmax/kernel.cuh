#ifndef INFINI_OPS_CUDA_ARGMAX_KERNEL_CUH_
#define INFINI_OPS_CUDA_ARGMAX_KERNEL_CUH_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cub/device/device_reduce.cuh>

namespace infini::ops::cuda_argmax_detail {

constexpr std::size_t Align256(std::size_t size) {
  return (size + 255) & ~std::size_t{255};
}

template <typename T>
std::size_t WorkspaceSize(std::size_t numel) {
  std::size_t cub_workspace_size = 0;
  auto error = cub::DeviceReduce::ArgMax(
      nullptr, cub_workspace_size, static_cast<const T*>(nullptr),
      static_cast<cub::KeyValuePair<int, T>*>(nullptr),
      static_cast<int>(numel));
  assert(error == 0 && "`CudaArgmax` failed to query CUB workspace");

  return Align256(sizeof(cub::KeyValuePair<int, T>)) + cub_workspace_size;
}

template <typename T>
__global__ void StoreIndex(int64_t* out,
                           const cub::KeyValuePair<int, T>* result) {
  *out = static_cast<int64_t>(result->key);
}

template <typename T, typename Stream>
void Launch(void* workspace, std::size_t workspace_size, const T* input,
            std::size_t numel, int64_t* out, Stream stream) {
  auto* result = static_cast<cub::KeyValuePair<int, T>*>(workspace);
  auto* cub_workspace = static_cast<char*>(workspace) +
                        Align256(sizeof(cub::KeyValuePair<int, T>));
  auto cub_workspace_size =
      workspace_size - Align256(sizeof(cub::KeyValuePair<int, T>));
  auto error =
      cub::DeviceReduce::ArgMax(cub_workspace, cub_workspace_size, input,
                                result, static_cast<int>(numel), stream);
  assert(error == 0 && "`CudaArgmax` CUB reduction failed");
  StoreIndex<<<1, 1, 0, stream>>>(out, result);
}

}  // namespace infini::ops::cuda_argmax_detail

#endif  // INFINI_OPS_CUDA_ARGMAX_KERNEL_CUH_
