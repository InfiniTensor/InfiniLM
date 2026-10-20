#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "config.h"
#include "runtime_api.h"
#include "tensor.h"

int main() {
  using namespace infini::ops;

#ifndef WITH_NVIDIA
  std::cout << "This example requires a `WITH_NVIDIA` build.\n";
  return 0;
#else
  constexpr auto m{2};
  constexpr auto k{3};
  constexpr auto n{4};
  constexpr std::size_t workspace_size_in_bytes = 8 * 1024 * 1024;

  std::vector<Tensor::Size> a_shape{m, k};
  std::vector<Tensor::Size> b_shape{k, n};
  std::vector<Tensor::Size> c_shape{m, n};

  const auto a_num_elements{std::accumulate(a_shape.cbegin(), a_shape.cend(), 1,
                                            std::multiplies<int>())};
  const auto b_num_elements{std::accumulate(b_shape.cbegin(), b_shape.cend(), 1,
                                            std::multiplies<int>())};
  const auto c_num_elements{std::accumulate(c_shape.cbegin(), c_shape.cend(), 1,
                                            std::multiplies<int>())};

  std::vector<float> a_vec(a_num_elements);
  std::vector<float> b_vec(b_num_elements);
  std::vector<float> c0_vec(c_num_elements, 0.0f);
  std::vector<float> c1_vec(c_num_elements, 0.0f);

  std::iota(a_vec.begin(), a_vec.end(), 0.0f);
  std::iota(b_vec.begin(), b_vec.end(), 0.0f);

  Device host_dev{Device::Type::kCpu};
  Device device_dev{DefaultRuntimeUtils::kDeviceType};

  Tensor a_host{a_vec.data(), a_shape, host_dev};
  Tensor b_host{b_vec.data(), b_shape, host_dev};
  Tensor c0_host{c0_vec.data(), c_shape, host_dev};
  Tensor c1_host{c1_vec.data(), c_shape, host_dev};

  const auto a_size{a_num_elements * kDataTypeToSize.at(a_host.dtype())};
  const auto b_size{b_num_elements * kDataTypeToSize.at(b_host.dtype())};
  const auto c_size{c_num_elements * kDataTypeToSize.at(c0_host.dtype())};

  void *a_ptr, *b_ptr, *c0_ptr, *c1_ptr, *workspace_ptr;

  DefaultRuntimeUtils::Malloc(&a_ptr, a_size);
  DefaultRuntimeUtils::Malloc(&b_ptr, b_size);
  DefaultRuntimeUtils::Malloc(&c0_ptr, c_size);
  DefaultRuntimeUtils::Malloc(&c1_ptr, c_size);
  DefaultRuntimeUtils::Malloc(&workspace_ptr, workspace_size_in_bytes);

  DefaultRuntimeUtils::Memcpy(a_ptr, a_vec.data(), a_size,
                              DefaultRuntimeUtils::kMemcpyHostToDevice);
  DefaultRuntimeUtils::Memcpy(b_ptr, b_vec.data(), b_size,
                              DefaultRuntimeUtils::kMemcpyHostToDevice);
  DefaultRuntimeUtils::Memset(c0_ptr, 0, c_size);
  DefaultRuntimeUtils::Memset(c1_ptr, 0, c_size);

  Tensor a_device{a_ptr, a_host.shape(), a_host.dtype(), device_dev,
                  a_host.strides()};
  Tensor b_device{b_ptr, b_host.shape(), b_host.dtype(), device_dev,
                  b_host.strides()};
  Tensor c0_device{c0_ptr, c0_host.shape(), c0_host.dtype(), device_dev,
                   c0_host.strides()};
  Tensor c1_device{c1_ptr, c1_host.shape(), c1_host.dtype(), device_dev,
                   c1_host.strides()};

  Handle handle;
  handle.set_workspace(workspace_ptr);
  handle.set_workspace_size_in_bytes(workspace_size_in_bytes);

  Config cublas_config;
  Config cublaslt_config;
  cublaslt_config.set_implementation_index(1);

  const auto c0_make_device{Tensor{c0_ptr, c0_host.shape(), c0_host.dtype(),
                                   device_dev, c0_host.strides()}};
  const auto c1_make_device{Tensor{c1_ptr, c1_host.shape(), c1_host.dtype(),
                                   device_dev, c1_host.strides()}};
  auto op0 = Gemm::Make(cublas_config, a_device, b_device, c0_make_device);
  auto op1 = Gemm::Make(cublaslt_config, a_device, b_device, c1_make_device);
  const auto* op0_ptr = op0.get();
  const auto* op1_ptr = op1.get();

  Gemm::Call(handle, cublas_config, a_device, b_device, c0_device);
  Gemm::Call(handle, cublaslt_config, a_device, b_device, c1_device);

  DefaultRuntimeUtils::Memcpy(c0_vec.data(), c0_ptr, c_size,
                              DefaultRuntimeUtils::kMemcpyDeviceToHost);
  DefaultRuntimeUtils::Memcpy(c1_vec.data(), c1_ptr, c_size,
                              DefaultRuntimeUtils::kMemcpyDeviceToHost);

  DefaultRuntimeUtils::Free(workspace_ptr);
  DefaultRuntimeUtils::Free(c1_ptr);
  DefaultRuntimeUtils::Free(c0_ptr);
  DefaultRuntimeUtils::Free(b_ptr);
  DefaultRuntimeUtils::Free(a_ptr);

  float max_abs_diff{0.0f};
  for (std::size_t i = 0; i < c0_vec.size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(c0_vec[i] - c1_vec[i]));
  }

  std::cout << "implementation 0 type: " << typeid(*op0_ptr).name() << "\n";
  std::cout << "implementation 1 type: " << typeid(*op1_ptr).name() << "\n";
  std::cout << "different implementation types: "
            << (typeid(*op0_ptr) != typeid(*op1_ptr) ? "yes" : "no") << "\n";
  std::cout << "cuBLAS (implementation 0): " << c0_host.ToString() << "\n";
  std::cout << "cuBLASLt (implementation 1): " << c1_host.ToString() << "\n";
  std::cout << "max_abs_diff: " << std::setprecision(8) << max_abs_diff << "\n";

  return max_abs_diff == 0.0f && typeid(*op0_ptr) != typeid(*op1_ptr) ? 0 : 1;
#endif
}
