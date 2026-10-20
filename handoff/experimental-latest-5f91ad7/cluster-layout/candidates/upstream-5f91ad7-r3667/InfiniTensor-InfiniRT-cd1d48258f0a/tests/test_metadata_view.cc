#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/metadata_view.h"
#include "common/small_vector.h"
#include "test_helper.h"

namespace {

using MetadataView = infini::rt::detail::MetadataView<std::size_t>;
using SmallVector = infini::rt::detail::SmallVector<std::size_t, 4>;
using infini::rt::test::TestContext;

static_assert(std::is_trivially_copyable_v<MetadataView>);
static_assert(std::is_nothrow_copy_constructible_v<MetadataView>);
static_assert(std::is_nothrow_copy_assignable_v<MetadataView>);
static_assert(std::is_same_v<MetadataView::reference, const std::size_t&>);
static_assert(std::is_same_v<MetadataView::pointer, const std::size_t*>);
static_assert(std::is_same_v<MetadataView::iterator, const std::size_t*>);
static_assert(std::is_same_v<decltype(std::declval<MetadataView&>().data()),
                             const std::size_t*>);
static_assert(std::is_same_v<decltype(std::declval<MetadataView&>().front()),
                             const std::size_t&>);
static_assert(std::is_same_v<decltype(std::declval<MetadataView&>().back()),
                             const std::size_t&>);
static_assert(std::is_same_v<decltype(std::declval<MetadataView&>()[0]),
                             const std::size_t&>);
static_assert(std::is_same_v<decltype(std::declval<MetadataView&>().begin()),
                             const std::size_t*>);

void TestEmptyView(TestContext* context) {
  const MetadataView empty;
  context->Expect(empty.empty(), "A default `MetadataView` should be empty.");
  context->ExpectEqual(empty.size(), std::size_t{0},
                       "A default `MetadataView` should have size zero.");
  context->Expect(empty.data() == nullptr,
                  "A default `MetadataView` should have null data.");
  context->Expect(empty.begin() == nullptr,
                  "A default `MetadataView` should have a null begin.");
  context->Expect(empty.end() == nullptr,
                  "A default `MetadataView` should have a null end.");
  context->Expect(empty.cbegin() == empty.begin(),
                  "Empty `begin()` and `cbegin()` should agree.");
  context->Expect(empty.cend() == empty.end(),
                  "Empty `end()` and `cend()` should agree.");

  const std::array<std::size_t, 1> storage{7};
  const MetadataView empty_at_data{storage.data(), 0};
  context->Expect(empty_at_data.begin() == storage.data(),
                  "An empty `MetadataView` should preserve non-null data.");
  context->Expect(empty_at_data.end() == storage.data(),
                  "An empty `MetadataView` should end at its data pointer.");
}

void TestAccessors(TestContext* context) {
  std::array<std::size_t, 3> storage{2, 4, 6};
  const MetadataView view{storage.data(), storage.size()};

  context->Expect(!view.empty(),
                  "A `MetadataView` with values should not be empty.");
  context->ExpectEqual(view.size(), storage.size(),
                       "`MetadataView` should report its size.");
  context->Expect(view.data() == storage.data(),
                  "`MetadataView` should preserve its data pointer.");
  context->ExpectEqual(view.front(), std::size_t{2},
                       "`front()` should expose the first value.");
  context->ExpectEqual(view[1], std::size_t{4},
                       "`operator[]` should expose the selected value.");
  context->ExpectEqual(view.back(), std::size_t{6},
                       "`back()` should expose the final value.");
  context->Expect(view.begin() == view.cbegin(),
                  "`begin()` and `cbegin()` should agree.");
  context->Expect(view.end() == view.cend(),
                  "`end()` and `cend()` should agree.");
  context->Expect(view.end() == storage.data() + storage.size(),
                  "`end()` should follow the final value.");

  storage[1] = 8;
  context->ExpectEqual(view[1], std::size_t{8},
                       "`MetadataView` should observe its referenced storage.");
}

void TestViewEquality(TestContext* context) {
  const std::array<std::size_t, 3> values{1, 2, 3};
  const std::array<unsigned int, 3> equal_values{1, 2, 3};
  const std::array<unsigned int, 3> different_values{1, 2, 4};
  const MetadataView view{values.data(), values.size()};
  const infini::rt::detail::MetadataView<unsigned int> equal_view{
      equal_values.data(), equal_values.size()};
  const infini::rt::detail::MetadataView<unsigned int> different_view{
      different_values.data(), different_values.size()};

  context->Expect(view == equal_view && equal_view == view,
                  "Compatible `MetadataView` types should compare by value.");
  context->Expect(view != different_view && different_view != view,
                  "`MetadataView` should detect unequal values.");
}

void TestRangeEquality(TestContext* context) {
  const std::array<std::size_t, 3> values{1, 2, 3};
  const MetadataView view{values.data(), values.size()};

  const std::array<std::size_t, 3> equal_array{1, 2, 3};
  const std::array<std::size_t, 3> different_array{1, 2, 4};
  context->Expect(view == equal_array && equal_array == view,
                  "`MetadataView` and `std::array` should compare by value.");
  context->Expect(
      view != different_array && different_array != view,
      "`MetadataView` and `std::array` should detect unequal values.");

  const std::vector<std::size_t> equal_vector{1, 2, 3};
  const std::vector<std::size_t> shorter_vector{1, 2};
  context->Expect(view == equal_vector && equal_vector == view,
                  "`MetadataView` and `std::vector` should compare by value.");
  context->Expect(view != shorter_vector && shorter_vector != view,
                  "`MetadataView` should detect a different range size.");

  const SmallVector equal_small_vector{1, 2, 3};
  const SmallVector different_small_vector{1, 2, 4};
  context->Expect(view == equal_small_vector && equal_small_vector == view,
                  "`MetadataView` and `SmallVector` should compare by value.");
  context->Expect(
      view != different_small_vector && different_small_vector != view,
      "`MetadataView` and `SmallVector` should detect unequal values.");
}

}  // namespace

int main() {
  TestContext context;

  TestEmptyView(&context);
  TestAccessors(&context);
  TestViewEquality(&context);
  TestRangeEquality(&context);

  return context.ExitCode();
}
