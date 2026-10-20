#include <infini/rt.h>

#if defined(__has_include)
#if __has_include(<infini/rt/detail/common/small_vector.h>)
#include <infini/rt/detail/common/small_vector.h>
#define INFINI_RT_HAS_SMALL_VECTOR 1
#endif
#endif

#ifndef INFINI_RT_HAS_SMALL_VECTOR
#define INFINI_RT_HAS_SMALL_VECTOR 0
#endif

#include <array>
#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::TensorView;

constexpr std::size_t kIterations = 200000;

struct VectorTensorLike {
  void* data_value;

  std::vector<std::size_t> shape_value;

  DataType dtype_value;

  Device device_value;

  std::vector<std::ptrdiff_t> strides_value;

  void* data() const { return data_value; }

  const std::vector<std::size_t>& shape() const { return shape_value; }

  DataType dtype() const { return dtype_value; }

  Device device() const { return device_value; }

  const std::vector<std::ptrdiff_t>& strides() const { return strides_value; }
};

INFINI_RT_NOINLINE std::size_t ConsumeTensorView(TensorView tensor) {
  perf::DoNotOptimize(tensor.data());
  return tensor.ndim() + tensor.size(0) +
         static_cast<std::size_t>(tensor.stride(0));
}

INFINI_RT_NOINLINE std::size_t ConsumeTensorMetadata(const TensorView& tensor) {
  const auto rank = tensor.ndim();
  std::size_t result = rank;

  for (TensorView::Index axis = 0; axis < static_cast<TensorView::Index>(rank);
       ++axis) {
    result += tensor.size(axis);
    result += static_cast<std::size_t>(tensor.stride(axis));
  }

  return result;
}

template <std::size_t rank>
std::array<TensorView::Size, rank> MakeShape() {
  std::array<TensorView::Size, rank> shape{};
  shape.fill(2);
  return shape;
}

template <std::size_t rank>
std::array<TensorView::Stride, rank> MakeStrides(
    const std::array<TensorView::Size, rank>& shape) {
  std::array<TensorView::Stride, rank> strides{};
  TensorView::Stride stride = 1;

  for (std::size_t i = rank; i > 0; --i) {
    strides[i - 1] = stride;
    stride *= static_cast<TensorView::Stride>(shape[i - 1]);
  }

  return strides;
}

template <std::size_t rank>
TensorView MakeInitializerListTensor(float* data, const Device& device);

template <>
TensorView MakeInitializerListTensor<1>(float* data, const Device& device) {
  return TensorView{data, {2}, DataType::kFloat32, device, {1}};
}

template <>
TensorView MakeInitializerListTensor<2>(float* data, const Device& device) {
  return TensorView{data, {2, 2}, DataType::kFloat32, device, {2, 1}};
}

template <>
TensorView MakeInitializerListTensor<4>(float* data, const Device& device) {
  return TensorView{
      data, {2, 2, 2, 2}, DataType::kFloat32, device, {8, 4, 2, 1}};
}

template <>
TensorView MakeInitializerListTensor<5>(float* data, const Device& device) {
  return TensorView{
      data, {2, 2, 2, 2, 2}, DataType::kFloat32, device, {16, 8, 4, 2, 1}};
}

template <>
TensorView MakeInitializerListTensor<8>(float* data, const Device& device) {
  return TensorView{data,
                    {2, 2, 2, 2, 2, 2, 2, 2},
                    DataType::kFloat32,
                    device,
                    {128, 64, 32, 16, 8, 4, 2, 1}};
}

template <>
TensorView MakeInitializerListTensor<9>(float* data, const Device& device) {
  return TensorView{data,
                    {2, 2, 2, 2, 2, 2, 2, 2, 2},
                    DataType::kFloat32,
                    device,
                    {256, 128, 64, 32, 16, 8, 4, 2, 1}};
}

template <std::size_t rank>
void RunRankBenchmarks(float* data, const Device& device) {
  const auto shape_values = MakeShape<rank>();
  const auto stride_values = MakeStrides(shape_values);
  const TensorView::Shape shape{shape_values.begin(), shape_values.end()};
  const TensorView::Strides strides{stride_values.begin(), stride_values.end()};
  const VectorTensorLike tensor_like{
      data,
      {shape_values.begin(), shape_values.end()},
      DataType::kFloat32,
      device,
      {stride_values.begin(), stride_values.end()}};
  const TensorView source{data, shape, DataType::kFloat32, device, strides};
  const auto params = std::vector<perf::Param>{perf::NumberParam("ndim", rank)};

  perf::RunBenchmark("perf_tensor_view.construct_lvalue_explicit", params,
                     kIterations, "ns", [&] {
                       TensorView tensor{data, shape, DataType::kFloat32,
                                         device, strides};
                       perf::DoNotOptimize(tensor);
                     });

  perf::RunBenchmark(
      "perf_tensor_view.construct_rvalue_explicit", params, kIterations, "ns",
      [&] {
        TensorView tensor{
            data, TensorView::Shape{shape_values.begin(), shape_values.end()},
            DataType::kFloat32, device,
            TensorView::Strides{stride_values.begin(), stride_values.end()}};
        perf::DoNotOptimize(tensor);
      });

  perf::RunBenchmark(
      "perf_tensor_view.construct_default_strides", params, kIterations, "ns",
      [&] {
        TensorView tensor{
            data, TensorView::Shape{shape_values.begin(), shape_values.end()},
            DataType::kFloat32, device};
        perf::DoNotOptimize(tensor);
      });

  perf::RunBenchmark("perf_tensor_view.construct_initializer_list", params,
                     kIterations, "ns", [&] {
                       const auto tensor =
                           MakeInitializerListTensor<rank>(data, device);
                       perf::DoNotOptimize(tensor);
                     });

  perf::RunBenchmark("perf_tensor_view.construct_tensor_like", params,
                     kIterations, "ns", [&] {
                       TensorView tensor{tensor_like};
                       perf::DoNotOptimize(tensor);
                     });

  perf::RunBenchmark("perf_tensor_view.copy", params, kIterations, "ns", [&] {
    TensorView tensor{source};
    perf::DoNotOptimize(tensor);
  });

  perf::RunBenchmark("perf_tensor_view.operator_index", params, kIterations,
                     "ns", [&] {
                       const auto tensor = source[0];
                       perf::DoNotOptimize(tensor);
                     });

  perf::RunBenchmark("perf_tensor_view.pass_by_value", params, kIterations,
                     "ns", [&] {
                       const auto value = ConsumeTensorView(source);
                       perf::DoNotOptimize(value);
                     });

  perf::RunBenchmark("perf_tensor_view.metadata_access", params, kIterations,
                     "ns", [&] {
                       const auto value = ConsumeTensorMetadata(source);
                       perf::DoNotOptimize(value);
                     });

  perf::RunBenchmark("perf_tensor_view.numel", params, kIterations, "ns", [&] {
    const auto value = source.numel();
    perf::DoNotOptimize(value);
  });
}

void RunRank2Controls(float* data, const Device& device) {
  const TensorView::Shape shape{2, 2};
  const TensorView::Strides contiguous_strides{2, 1};
  const TensorView::Strides transposed_strides{1, 2};
  const TensorView contiguous{data, shape, DataType::kFloat32, device,
                              contiguous_strides};
  const TensorView transposed{data, shape, DataType::kFloat32, device,
                              transposed_strides};
  const auto params = std::vector<perf::Param>{perf::NumberParam("ndim", 2)};

  perf::RunBenchmark("perf_tensor_view.transpose", params, kIterations, "ns",
                     [&] {
                       const auto tensor = contiguous.T();
                       perf::DoNotOptimize(tensor);
                     });

  perf::RunBenchmark("perf_tensor_view.is_contiguous_true", params, kIterations,
                     "ns", [&] {
                       const auto value = contiguous.IsContiguous();
                       perf::DoNotOptimize(value);
                     });

  perf::RunBenchmark("perf_tensor_view.is_contiguous_false", params,
                     kIterations, "ns", [&] {
                       const auto value = transposed.IsContiguous();
                       perf::DoNotOptimize(value);
                     });

  perf::RunBenchmark("perf_tensor_view.hash", params, kIterations, "ns", [&] {
    const auto value = std::hash<TensorView>{}(contiguous);
    perf::DoNotOptimize(value);
  });
}

}  // namespace

int main() {
  std::cerr << "sizeof(TensorView)=" << sizeof(TensorView)
            << " sizeof(Shape)=" << sizeof(TensorView::Shape)
            << " sizeof(Strides)=" << sizeof(TensorView::Strides)
            << " sizeof(ShapeView)=" << sizeof(TensorView::ShapeView)
            << " sizeof(StridesView)=" << sizeof(TensorView::StridesView)
            << '\n';

#if INFINI_RT_HAS_SMALL_VECTOR
  std::cerr
      << "sizeof(SmallVector<Size, 4>)="
      << sizeof(infini::rt::detail::SmallVector<TensorView::Size, 4>)
      << " sizeof(SmallVector<Size, 8>)="
      << sizeof(infini::rt::detail::SmallVector<TensorView::Size, 8>)
      << " sizeof(ShapeStridesStorage<4>)="
      << sizeof(infini::rt::detail::ShapeStridesStorage<TensorView::Size,
                                                        TensorView::Stride, 4>)
      << " sizeof(ShapeStridesStorage<8>)="
      << sizeof(infini::rt::detail::ShapeStridesStorage<TensorView::Size,
                                                        TensorView::Stride, 8>)
      << '\n';
#endif

  std::array<float, 512> data{};
  const Device cpu_device{Device::Type::kCpu};

  RunRankBenchmarks<1>(data.data(), cpu_device);
  RunRankBenchmarks<2>(data.data(), cpu_device);
  RunRankBenchmarks<4>(data.data(), cpu_device);
  RunRankBenchmarks<5>(data.data(), cpu_device);
  RunRankBenchmarks<8>(data.data(), cpu_device);
  RunRankBenchmarks<9>(data.data(), cpu_device);
  RunRank2Controls(data.data(), cpu_device);

  return 0;
}
