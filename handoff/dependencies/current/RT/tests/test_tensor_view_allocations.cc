#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "test_helper.h"

namespace {

thread_local bool count_allocations = false;
thread_local std::size_t allocation_count = 0;

class AllocationScope {
 public:
  AllocationScope() {
    allocation_count = 0;
    count_allocations = true;
  }

  AllocationScope(const AllocationScope&) = delete;
  AllocationScope& operator=(const AllocationScope&) = delete;

  ~AllocationScope() { count_allocations = false; }

  std::size_t count() const { return allocation_count; }
};

template <typename Function>
std::size_t CountAllocations(Function&& function) {
  AllocationScope scope;
  std::forward<Function>(function)();
  return scope.count();
}

void ExpectAllocationCount(infini::rt::test::TestContext* context,
                           std::size_t actual, std::size_t expected,
                           const char* message) {
  context->ExpectEqual(actual, expected, message);
}

}  // namespace

void* operator new(std::size_t size) {
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
    if (count_allocations) {
      ++allocation_count;
    }
    return pointer;
  }
  throw std::bad_alloc{};
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* pointer) noexcept { std::free(pointer); }

void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }

void operator delete(void* pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
  ::operator delete[](pointer);
}

namespace {

using infini::rt::DataType;
using infini::rt::Device;
using infini::rt::TensorView;

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

void ExpectRankAllocationCount(infini::rt::test::TestContext* context,
                               std::size_t actual, std::size_t expected,
                               std::size_t rank, const char* message) {
  std::string full_message = "Rank-" + std::to_string(rank) + " ";
  full_message += message;
  context->ExpectEqual(actual, expected, full_message);
}

template <std::size_t rank>
std::array<TensorView::Size, rank> MakeShapeValues() {
  std::array<TensorView::Size, rank> shape{};
  shape.fill(2);
  return shape;
}

template <std::size_t rank>
std::array<TensorView::Stride, rank> MakeStrideValues() {
  std::array<TensorView::Stride, rank> strides{};
  TensorView::Stride stride = 1;
  for (std::size_t index = rank; index > 0; --index) {
    strides[index - 1] = stride;
    stride *= 2;
  }
  return strides;
}

template <std::size_t rank, std::size_t... indices>
std::size_t CountInitializerListConstructionAllocations(
    void* data, const std::array<TensorView::Size, rank>& shape,
    const std::array<TensorView::Stride, rank>& strides, const Device& device,
    std::index_sequence<indices...>) {
  return CountAllocations([&] {
    TensorView tensor{
        data, std::initializer_list<TensorView::Size>{shape[indices]...},
        DataType::kFloat32, device,
        std::initializer_list<TensorView::Stride>{strides[indices]...}};
    (void)tensor;
  });
}

template <std::size_t rank>
void TestConstructionAllocationsForRank(infini::rt::test::TestContext* context,
                                        void* data, const Device& device) {
  constexpr std::size_t kCombinedMetadataAllocationCount = rank <= 8 ? 0 : 1;
  constexpr std::size_t kRvalueMetadataAllocationCount = rank <= 8 ? 0 : 2;
  constexpr std::size_t kGeneratedMetadataAllocationCount = rank <= 8 ? 0 : 1;

  const auto shape_values = MakeShapeValues<rank>();
  const auto stride_values = MakeStrideValues<rank>();
  const TensorView::Shape shape{shape_values.begin(), shape_values.end()};
  const TensorView::Strides strides{stride_values.begin(), stride_values.end()};
  const VectorTensorLike tensor_like{
      data, std::vector<std::size_t>{shape_values.begin(), shape_values.end()},
      DataType::kFloat32, device,
      std::vector<std::ptrdiff_t>{stride_values.begin(), stride_values.end()}};

  ExpectRankAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, shape, DataType::kFloat32, device, strides};
        (void)tensor;
      }),
      kCombinedMetadataAllocationCount, rank,
      "lvalue shape and strides should have the expected allocation count.");

  ExpectRankAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{
            data, TensorView::Shape{shape_values.begin(), shape_values.end()},
            DataType::kFloat32, device,
            TensorView::Strides{stride_values.begin(), stride_values.end()}};
        (void)tensor;
      }),
      kRvalueMetadataAllocationCount, rank,
      "exact-type rvalue shape and strides should have the expected allocation "
      "count.");

  ExpectRankAllocationCount(
      context,
      CountInitializerListConstructionAllocations(
          data, shape_values, stride_values, device,
          std::make_index_sequence<rank>{}),
      kCombinedMetadataAllocationCount, rank,
      "initializer-list overload should have the expected allocation count.");

  ExpectRankAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{tensor_like};
        (void)tensor;
      }),
      kCombinedMetadataAllocationCount, rank,
      "vector-backed `TensorLike` should have the expected allocation count.");

  ExpectRankAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, shape, DataType::kFloat32, device};
        (void)tensor;
      }),
      kCombinedMetadataAllocationCount, rank,
      "ordinary default-stride construction should have the expected "
      "allocation count.");

  TensorView::Shape explicit_move_shape{shape_values.begin(),
                                        shape_values.end()};
  TensorView::Strides explicit_move_strides{stride_values.begin(),
                                            stride_values.end()};
  ExpectRankAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, std::move(explicit_move_shape),
                          DataType::kFloat32, device,
                          std::move(explicit_move_strides)};
        (void)tensor;
      }),
      0, rank, "moved exact explicit metadata should not allocate.");

  TensorView::Shape default_move_shape{shape_values.begin(),
                                       shape_values.end()};
  ExpectRankAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data, std::move(default_move_shape),
                          DataType::kFloat32, device};
        (void)tensor;
      }),
      kGeneratedMetadataAllocationCount, rank,
      "moved shape with generated default strides should have the expected "
      "allocation count.");
}

template <std::size_t... ranks>
void TestConstructionAllocationsForRanks(infini::rt::test::TestContext* context,
                                         void* data, const Device& device,
                                         std::index_sequence<ranks...>) {
  (TestConstructionAllocationsForRank<ranks>(context, data, device), ...);
}

void TestConstructionAllocationMatrix(infini::rt::test::TestContext* context) {
  std::array<float, 512> data{};
  const Device cpu{Device::Type::kCpu};

  TestConstructionAllocationsForRanks(context, data.data(), cpu,
                                      std::make_index_sequence<10>{});
}

void TestValueAndDerivedViewAllocations(
    infini::rt::test::TestContext* context) {
  std::array<float, 512> data{};
  const Device cpu{Device::Type::kCpu};
  const TensorView::Shape shape8{2, 2, 2, 2, 2, 2, 2, 2};
  const TensorView::Strides strides8{128, 64, 32, 16, 8, 4, 2, 1};
  const TensorView::Shape shape9{2, 2, 2, 2, 2, 2, 2, 2, 2};
  const TensorView::Strides strides9{256, 128, 64, 32, 16, 8, 4, 2, 1};
  const TensorView source8{data.data(), shape8, DataType::kFloat32, cpu,
                           strides8};
  const TensorView source9{data.data(), shape9, DataType::kFloat32, cpu,
                           strides9};

  ExpectAllocationCount(context, CountAllocations([&] {
                          TensorView copied{source8};
                          (void)copied;
                        }),
                        0, "Copying rank-8 metadata should stay inline.");
  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView copied{source9};
        (void)copied;
      }),
      1, "Copying rank-9 metadata should use one combined allocation.");

  TensorView move_source8{data.data(), shape8, DataType::kFloat32, cpu,
                          strides8};
  TensorView move_source9{data.data(), shape9, DataType::kFloat32, cpu,
                          strides9};

  ExpectAllocationCount(context, CountAllocations([&] {
                          TensorView moved{std::move(move_source8)};
                          (void)moved;
                        }),
                        0, "Moving rank-8 metadata should not allocate.");
  ExpectAllocationCount(context, CountAllocations([&] {
                          TensorView moved{std::move(move_source9)};
                          (void)moved;
                        }),
                        0,
                        "Moving rank-9 metadata should transfer heap storage.");

  ExpectAllocationCount(context, CountAllocations([&] {
                          TensorView indexed = source8[0];
                          (void)indexed;
                        }),
                        0, "Indexing rank-8 to rank-7 should stay inline.");
  ExpectAllocationCount(context, CountAllocations([&] {
                          TensorView indexed = source9[0];
                          (void)indexed;
                        }),
                        0, "Indexing rank-9 to rank-8 should stay inline.");

  const TensorView transpose_source{data.data(), TensorView::Shape{2, 2},
                                    DataType::kFloat32, cpu,
                                    TensorView::Strides{2, 1}};
  ExpectAllocationCount(context, CountAllocations([&] {
                          TensorView transposed = transpose_source.T();
                          (void)transposed;
                        }),
                        0, "Transposing rank-2 metadata should stay inline.");
}

void TestDefaultMetadataAllocations(infini::rt::test::TestContext* context) {
  std::array<float, 32> data{};
  const Device cpu{Device::Type::kCpu};
  const Device indexed_cpu{Device::Type::kCpu, 1};
  const TensorView::Shape expected_shape{2, 3};
  const TensorView::Strides expected_strides{3, 1};

  bool shape_only_metadata_is_default = false;
  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data.data(), TensorView::Shape{2, 3}};
        shape_only_metadata_is_default = tensor.shape() == expected_shape &&
                                         tensor.dtype() == DataType::kFloat32 &&
                                         tensor.device() == cpu &&
                                         tensor.strides() == expected_strides;
      }),
      0, "Rank-2 shape-only construction should stay inline.");
  context->Expect(shape_only_metadata_is_default,
                  "Shape-only construction should keep default metadata.");

  bool dtype_only_metadata_is_default = false;
  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data.data(), TensorView::Shape{2, 3},
                          DataType::kFloat64};
        dtype_only_metadata_is_default = tensor.shape() == expected_shape &&
                                         tensor.dtype() == DataType::kFloat64 &&
                                         tensor.device() == cpu &&
                                         tensor.strides() == expected_strides;
      }),
      0, "Rank-2 shape and dtype construction should stay inline.");
  context->Expect(
      dtype_only_metadata_is_default,
      "Shape and dtype construction should keep default device and strides.");

  bool device_only_metadata_is_default = false;
  ExpectAllocationCount(
      context, CountAllocations([&] {
        TensorView tensor{data.data(), TensorView::Shape{2, 3}, indexed_cpu};
        device_only_metadata_is_default =
            tensor.shape() == expected_shape &&
            tensor.dtype() == DataType::kFloat32 &&
            tensor.device() == indexed_cpu &&
            tensor.strides() == expected_strides;
      }),
      0, "Rank-2 shape and device construction should stay inline.");
  context->Expect(
      device_only_metadata_is_default,
      "Shape and device construction should keep default dtype and strides.");
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  TestConstructionAllocationMatrix(&context);
  TestValueAndDerivedViewAllocations(&context);
  TestDefaultMetadataAllocations(&context);

  return context.ExitCode();
}
