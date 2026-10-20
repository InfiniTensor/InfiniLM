#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::TensorView;

constexpr std::size_t kTensorVisitsPerSample = 262144;
constexpr std::array<std::size_t, 2> kTensorCounts = {8, 256};

volatile std::uintptr_t g_benchmark_sink = 0;

struct CacheKeyLike {
  std::size_t hash{0};

  std::vector<TensorView> tensors;

  std::size_t scalar_hash{0};
};

void HashCombine(std::size_t& seed, std::size_t value) {
  seed ^= std::hash<std::size_t>{}(value) +
          static_cast<std::size_t>(0x9e3779b9) + (seed << 6) + (seed >> 2);
}

void HashCombine(std::size_t& seed, const TensorView& value) {
  seed ^= std::hash<TensorView>{}(value) +
          static_cast<std::size_t>(0x9e3779b9) + (seed << 6) + (seed >> 2);
}

INFINI_RT_NOINLINE CacheKeyLike
BuildCacheKeyLike(const std::vector<TensorView>& inputs) {
  CacheKeyLike key;
  HashCombine(key.hash, inputs.size());
  for (const auto& input : inputs) {
    HashCombine(key.hash, input);
    key.tensors.push_back(input);
  }
  return key;
}

INFINI_RT_NOINLINE bool EqualCacheKeys(const CacheKeyLike& lhs,
                                       const CacheKeyLike& rhs) {
  if (lhs.scalar_hash != rhs.scalar_hash ||
      lhs.tensors.size() != rhs.tensors.size()) {
    return false;
  }

  const std::equal_to<TensorView> equal;
  for (std::size_t i = 0; i < lhs.tensors.size(); ++i) {
    if (!equal(lhs.tensors[i], rhs.tensors[i])) {
      return false;
    }
  }
  return true;
}

template <std::size_t rank>
TensorView::Shape MakeShape() {
  TensorView::Shape shape(rank);
  for (auto& size : shape) {
    size = 2;
  }
  return shape;
}

TensorView::Strides MakeStrides(const TensorView::Shape& shape) {
  TensorView::Strides strides(shape.size());
  TensorView::Stride stride = 1;

  for (std::size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= static_cast<TensorView::Stride>(shape[i - 1]);
  }

  return strides;
}

template <std::size_t rank>
std::vector<TensorView> MakeInputs(float* data, const Device& device,
                                   std::size_t tensor_count) {
  std::vector<TensorView> inputs;
  inputs.reserve(tensor_count);

  for (std::size_t i = 0; i < tensor_count; ++i) {
    auto shape = MakeShape<rank>();
    shape[0] += (i & 3);
    const auto strides = MakeStrides(shape);
    inputs.emplace_back(data, shape, DataType::kFloat32, device, strides);
  }
  return inputs;
}

template <std::size_t rank>
void RunFootprintBenchmarks(float* data, const Device& device) {
  for (const auto tensor_count : kTensorCounts) {
    const auto inputs = MakeInputs<rank>(data, device, tensor_count);
    const auto reference = BuildCacheKeyLike(inputs);
    const auto iterations = kTensorVisitsPerSample / tensor_count;
    const auto params = std::vector<perf::Param>{
        perf::NumberParam("ndim", rank),
        perf::NumberParam("tensor_count", tensor_count)};

    perf::RunBenchmark(
        "perf_tensor_view_footprint.cache_key_build_hit", params, iterations,
        "ns", [&] {
          const auto candidate = BuildCacheKeyLike(inputs);
          const bool equal = EqualCacheKeys(candidate, reference);
          g_benchmark_sink =
              static_cast<std::uintptr_t>(candidate.hash) ^
              reinterpret_cast<std::uintptr_t>(candidate.tensors.data()) ^
              static_cast<std::uintptr_t>(equal);
        });
  }
}

}  // namespace

int main() {
  std::cerr << "sizeof(TensorView)=" << sizeof(TensorView)
            << " sizeof(Shape)=" << sizeof(TensorView::Shape)
            << " sizeof(Strides)=" << sizeof(TensorView::Strides) << '\n';

  std::array<float, 1024> data{};
  const Device cpu_device{Device::Type::kCpu};

  RunFootprintBenchmarks<4>(data.data(), cpu_device);
  RunFootprintBenchmarks<8>(data.data(), cpu_device);

  return 0;
}
