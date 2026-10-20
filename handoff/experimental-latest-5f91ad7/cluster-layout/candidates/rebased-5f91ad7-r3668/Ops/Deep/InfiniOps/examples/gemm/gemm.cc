#include <iostream>
#include <numeric>
#include <variant>

#include "runtime_api.h"
#include "tensor.h"

int main() {
  using namespace infini::ops;

  constexpr auto m{2};
  constexpr auto k{3};
  constexpr auto n{4};

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
  std::vector<float> c_vec(c_num_elements);

  std::iota(a_vec.begin(), a_vec.end(), 0);
  std::iota(b_vec.begin(), b_vec.end(), 0);

  Device host_dev{Device::Type::kCpu};
  Device device_dev{DefaultRuntimeUtils::kDeviceType};

  Tensor a_host{a_vec.data(), a_shape, host_dev};
  Tensor b_host{b_vec.data(), b_shape, host_dev};
  Tensor c_host{c_vec.data(), c_shape, host_dev};

  const auto a_size{a_num_elements * kDataTypeToSize.at(a_host.dtype())};
  const auto b_size{b_num_elements * kDataTypeToSize.at(b_host.dtype())};
  const auto c_size{c_num_elements * kDataTypeToSize.at(c_host.dtype())};

  void *a_ptr, *b_ptr, *c_ptr;

  DefaultRuntimeUtils::Malloc(&a_ptr, a_size);
  DefaultRuntimeUtils::Malloc(&b_ptr, b_size);
  DefaultRuntimeUtils::Malloc(&c_ptr, c_size);

  DefaultRuntimeUtils::Memcpy(a_ptr, a_vec.data(), a_size,
                              DefaultRuntimeUtils::kMemcpyHostToDevice);
  DefaultRuntimeUtils::Memcpy(b_ptr, b_vec.data(), b_size,
                              DefaultRuntimeUtils::kMemcpyHostToDevice);
  DefaultRuntimeUtils::Memset(c_ptr, 0, c_size);

  Tensor a_device{a_ptr, a_host.shape(), a_host.dtype(), device_dev,
                  a_host.strides()};
  Tensor b_device{b_ptr, b_host.shape(), b_host.dtype(), device_dev,
                  b_host.strides()};
  Tensor c_device{c_ptr, c_host.shape(), c_host.dtype(), device_dev,
                  c_host.strides()};

  Gemm::Call(a_device, b_device, c_device);

  DefaultRuntimeUtils::Memcpy(c_vec.data(), c_ptr, c_size,
                              DefaultRuntimeUtils::kMemcpyDeviceToHost);
  DefaultRuntimeUtils::Free(a_ptr);
  DefaultRuntimeUtils::Free(b_ptr);
  DefaultRuntimeUtils::Free(c_ptr);

  std::cout << "A: " << a_host.ToString() << "\n";
  std::cout << "B: " << b_host.ToString() << "\n";
  std::cout << "C: " << c_host.ToString() << "\n";

  return 0;
}
