#include <array>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/small_vector.h"
#include "test_helper.h"

namespace {

thread_local bool count_allocations = false;
thread_local std::size_t allocation_count = 0;
thread_local bool count_deallocations = false;
thread_local std::size_t deallocation_count = 0;

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

class DeallocationScope {
 public:
  DeallocationScope() {
    deallocation_count = 0;
    count_deallocations = true;
  }

  DeallocationScope(const DeallocationScope&) = delete;

  DeallocationScope& operator=(const DeallocationScope&) = delete;

  ~DeallocationScope() { count_deallocations = false; }

  std::size_t count() const { return deallocation_count; }
};

template <typename Function>
std::size_t CountAllocations(Function&& function) {
  AllocationScope scope;
  std::forward<Function>(function)();
  return scope.count();
}

template <typename Function>
std::size_t CountDeallocations(Function&& function) {
  DeallocationScope scope;
  std::forward<Function>(function)();
  return scope.count();
}

}  // namespace

void* operator new(std::size_t size) {
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
    if (count_allocations) ++allocation_count;
    return pointer;
  }
  throw std::bad_alloc{};
}

void* operator new[](std::size_t size) { return ::operator new(size); }

void operator delete(void* pointer) noexcept {
  if (pointer != nullptr && count_deallocations) ++deallocation_count;
  std::free(pointer);
}

void operator delete[](void* pointer) noexcept { ::operator delete(pointer); }

void operator delete(void* pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
  ::operator delete[](pointer);
}

namespace {

using Inline4 = infini::rt::detail::SmallVector<std::size_t, 4>;
using Inline8 = infini::rt::detail::SmallVector<std::size_t, 8>;
using HeapAllocation = Inline4::HeapAllocation;
using infini::rt::test::TestContext;

static_assert(std::is_copy_constructible_v<Inline4>);
static_assert(std::is_move_constructible_v<Inline4>);
static_assert(std::is_copy_assignable_v<Inline4>);
static_assert(std::is_move_assignable_v<Inline4>);
static_assert(
    std::is_constructible_v<Inline4, std::size_t, const std::size_t&>);
static_assert(!std::is_copy_constructible_v<HeapAllocation>);
static_assert(!std::is_copy_assignable_v<HeapAllocation>);
static_assert(std::is_nothrow_move_constructible_v<HeapAllocation>);
static_assert(std::is_nothrow_move_assignable_v<HeapAllocation>);

template <std::size_t inline_capacity>
void ExpectValues(
    TestContext* context,
    const infini::rt::detail::SmallVector<std::size_t, inline_capacity>& actual,
    std::initializer_list<std::size_t> expected, std::string_view message) {
  context->ExpectEqual(std::vector<std::size_t>(actual.begin(), actual.end()),
                       std::vector<std::size_t>(expected), message);
}

void ExpectHeapValues(TestContext* context, const HeapAllocation& actual,
                      std::initializer_list<std::size_t> expected,
                      std::string_view message) {
  context->ExpectEqual(
      std::vector<std::size_t>(actual.data(), actual.data() + actual.size()),
      std::vector<std::size_t>(expected), message);
}

void TestConstruction(TestContext* context) {
  Inline4 empty;
  context->Expect(empty.empty(), "A default `SmallVector` should be empty.");
  context->ExpectEqual(empty.size(), std::size_t{0},
                       "A default `SmallVector` should have size zero.");
  context->ExpectEqual(
      empty.capacity(), std::size_t{4},
      "A default `SmallVector` should expose inline capacity.");

  const std::size_t* null_range = nullptr;
  Inline4 empty_pointer_range{null_range, null_range};
  context->Expect(
      empty_pointer_range.empty(),
      "An empty null pointer range should construct an empty `SmallVector`.");

  Inline4 counted(3);
  ExpectValues(context, counted, {0, 0, 0},
               "The count constructor should value-initialize elements.");

  Inline4 empty_filled(0, 7);
  context->Expect(empty_filled.empty(),
                  "A zero-count fill constructor should be empty.");

  Inline4 inline_filled;
  const std::size_t inline_fill_allocations =
      CountAllocations([&] { inline_filled = Inline4(4, 7); });
  ExpectValues(context, inline_filled, {7, 7, 7, 7},
               "The fill constructor should initialize inline elements.");
  context->ExpectEqual(inline_fill_allocations, std::size_t{0},
                       "The inline fill constructor should not allocate.");
  context->ExpectEqual(
      inline_filled.capacity(), std::size_t{4},
      "The inline fill constructor should preserve inline storage.");

  Inline4 overflow_filled;
  const std::size_t overflow_fill_allocations =
      CountAllocations([&] { overflow_filled = Inline4(5, 9); });
  ExpectValues(context, overflow_filled, {9, 9, 9, 9, 9},
               "The fill constructor should initialize overflow elements.");
  context->ExpectEqual(overflow_fill_allocations, std::size_t{1},
                       "The overflow fill constructor should allocate once.");
  context->ExpectEqual(
      overflow_filled.capacity(), std::size_t{5},
      "The overflow fill constructor should allocate exact storage.");

  Inline4 inline_values{1, 2, 3, 4};
  context->ExpectEqual(inline_values.capacity(), std::size_t{4},
                       "Inline values should keep inline storage.");

  Inline4 overflow_values{1, 2, 3, 4, 5};
  context->Expect(overflow_values.capacity() >= 5,
                  "Overflow values should use sufficient heap storage.");

  const std::array<std::size_t, 3> iterator_source{2, 4, 6};
  Inline4 iterator_range(iterator_source.begin(), iterator_source.end());
  ExpectValues(context, iterator_range, {2, 4, 6},
               "The iterator-range constructor should preserve values.");

  std::istringstream input_stream{"5 7 9"};
  Inline4 input_range(std::istream_iterator<std::size_t>{input_stream},
                      std::istream_iterator<std::size_t>{});
  ExpectValues(context, input_range, {5, 7, 9},
               "The input-iterator constructor should consume the range "
               "once.");

  const std::vector<std::size_t> container_source{3, 6, 9};
  Inline4 container_range(container_source);
  ExpectValues(context, container_range, {3, 6, 9},
               "The container constructor should preserve values.");

  Inline8 wider_inline{1, 2, 3, 4, 5, 6, 7, 8};
  context->ExpectEqual(wider_inline.capacity(), std::size_t{8},
                       "`Inline8` should expose and use its inline capacity.");
  ExpectValues(context, wider_inline, {1, 2, 3, 4, 5, 6, 7, 8},
               "`Inline8` should preserve inline values.");
}

void TestAccessorsAndIterators(TestContext* context) {
  Inline4 values{1, 2, 3};
  context->ExpectEqual(values.size(), std::size_t{3},
                       "`SmallVector` should report its size.");
  context->Expect(!values.empty(),
                  "`SmallVector` with values should not be empty.");
  context->Expect(
      values.data() == values.begin(),
      "Mutable `data()` and `begin()` should identify the first value.");
  context->Expect(values.end() == values.data() + values.size(),
                  "Mutable `end()` should follow the final value.");

  values.front() = 4;
  values[1] = 5;
  values.back() = 6;
  context->ExpectEqual(*values.begin(), std::size_t{4},
                       "Mutable `begin()` should expose the first value.");
  context->ExpectEqual(values.data()[1], std::size_t{5},
                       "Mutable `data()` should expose indexed values.");
  context->ExpectEqual(*(values.end() - 1), std::size_t{6},
                       "Mutable `end()` should delimit the final value.");

  const Inline4& const_values = values;
  context->Expect(
      const_values.data() == const_values.begin(),
      "Const `data()` and `begin()` should identify the first value.");
  context->Expect(const_values.begin() == const_values.cbegin(),
                  "Const `begin()` and `cbegin()` should agree.");
  context->Expect(const_values.end() == const_values.cend(),
                  "Const `end()` and `cend()` should agree.");
  context->ExpectEqual(const_values.front(), std::size_t{4},
                       "Const `front()` should expose the first value.");
  context->ExpectEqual(const_values[1], std::size_t{5},
                       "Const indexing should expose indexed values.");
  context->ExpectEqual(const_values.back(), std::size_t{6},
                       "Const `back()` should expose the final value.");
}

void TestEquality(TestContext* context) {
  const Inline4 values{1, 2, 3};
  const std::vector<std::size_t> equal_values{1, 2, 3};
  const std::vector<std::size_t> different_values{1, 2, 4};

  context->Expect(values == equal_values,
                  "`SmallVector` should compare equal to `std::vector`.");
  context->Expect(equal_values == values,
                  "`std::vector` should compare equal to `SmallVector`.");
  context->Expect(values != different_values,
                  "`SmallVector` should compare unequal to `std::vector`.");
  context->Expect(different_values != values,
                  "`std::vector` should compare unequal to `SmallVector`.");

  const Inline8 wider_equal{1, 2, 3};
  const Inline8 wider_different{1, 2, 4};
  context->Expect(values == wider_equal && wider_equal == values,
                  "Different inline capacities should compare by value.");
  context->Expect(values != wider_different && wider_different != values,
                  "Different inline capacities should detect unequal values.");
}

void TestMutation(TestContext* context) {
  Inline4 cleared{1, 2, 3, 4, 5};
  cleared.clear();
  context->Expect(cleared.empty(), "`clear()` should remove every value.");
  context->ExpectEqual(cleared.size(), std::size_t{0},
                       "`clear()` should reset the size to zero.");

  Inline4 reserved{1, 2, 3};
  reserved.reserve(12);
  context->Expect(reserved.capacity() >= 12,
                  "`reserve()` should provide the requested capacity.");
  ExpectValues(context, reserved, {1, 2, 3},
               "`reserve()` should preserve existing values.");
  const std::size_t reserved_capacity = reserved.capacity();
  reserved.reserve(6);
  context->ExpectEqual(reserved.capacity(), reserved_capacity,
                       "`reserve()` should not shrink existing capacity.");

  Inline4 resized{1, 2};
  resized.resize(5);
  ExpectValues(context, resized, {1, 2, 0, 0, 0},
               "Growing `resize()` should value-initialize new elements.");
  context->Expect(resized.capacity() >= 5,
                  "Growing `resize()` should provide sufficient capacity.");
  resized.resize(1);
  ExpectValues(context, resized, {1},
               "Shrinking `resize()` should preserve the retained prefix.");

  Inline4 pushed;
  std::vector<std::size_t> pushed_expected;
  std::size_t previous_capacity = pushed.capacity();
  std::size_t growth_count = 0;
  for (std::size_t value = 0; value < 32; ++value) {
    pushed.push_back(value);
    pushed_expected.push_back(value);
    if (pushed.capacity() != previous_capacity) {
      const std::size_t new_capacity = pushed.capacity();
      context->Expect(new_capacity > previous_capacity,
                      "Repeated `push_back()` should increase capacity.");
      if (new_capacity > previous_capacity) {
        context->Expect(
            new_capacity - previous_capacity >= previous_capacity / 2,
            "Repeated `push_back()` should grow capacity multiplicatively.");
      }
      previous_capacity = new_capacity;
      ++growth_count;
    }
  }
  context->ExpectEqual(pushed.size(), pushed_expected.size(),
                       "Repeated `push_back()` should update the size.");
  context->Expect(pushed.capacity() >= pushed.size(),
                  "Repeated `push_back()` should provide sufficient capacity.");
  context->Expect(growth_count >= 2,
                  "Repeated `push_back()` should exercise multiple heap growth "
                  "steps.");
  context->Expect(pushed == pushed_expected,
                  "Repeated `push_back()` should preserve every value.");

  Inline4 assigned;
  assigned.assign({4, 5, 6});
  ExpectValues(context, assigned, {4, 5, 6},
               "Initializer-list `assign()` should replace values.");
  const std::vector<std::size_t> range_values{8, 6, 4, 2, 0};
  assigned.assign(range_values.begin(), range_values.end());
  context->Expect(assigned == range_values,
                  "Iterator-range `assign()` should replace values.");

  std::istringstream assign_input_stream{"9 7 5"};
  Inline4 input_assigned;
  input_assigned.assign(std::istream_iterator<std::size_t>{assign_input_stream},
                        std::istream_iterator<std::size_t>{});
  ExpectValues(context, input_assigned, {9, 7, 5},
               "Input-iterator `assign()` should consume the range once.");

  Inline4 heap_to_inline{1, 2, 3, 4, 5};
  heap_to_inline.assign({7, 8});
  ExpectValues(context, heap_to_inline, {7, 8},
               "A small assignment should replace overflow values.");
  context->ExpectEqual(heap_to_inline.capacity(), std::size_t{4},
                       "Assigning a small range should restore inline "
                       "storage.");

  Inline4 inline_to_heap{1, 2};
  inline_to_heap.assign({1, 2, 3, 4, 5});
  ExpectValues(context, inline_to_heap, {1, 2, 3, 4, 5},
               "An overflow assignment should replace inline values.");
  context->Expect(inline_to_heap.capacity() >= 5,
                  "Assigning an overflow range should use heap storage.");
}

void TestHeapRelease(TestContext* context) {
  Inline4 inline_values{1, 2, 3};
  std::size_t* const inline_data = inline_values.data();
  HeapAllocation inline_allocation;
  const std::size_t inline_release_allocations = CountAllocations(
      [&] { inline_allocation = inline_values.ReleaseHeap(); });
  context->ExpectEqual(inline_release_allocations, std::size_t{0},
                       "Releasing inline storage should not allocate.");
  context->Expect(inline_allocation.empty(),
                  "Releasing inline storage should return an empty owner.");
  context->Expect(inline_values.data() == inline_data,
                  "Releasing inline storage should preserve its address.");
  ExpectValues(context, inline_values, {1, 2, 3},
               "Releasing inline storage should preserve its values.");

  Inline4 overflow_values{1, 2, 3, 4, 5};
  overflow_values.reserve(12);
  std::size_t* const overflow_data = overflow_values.data();
  const std::size_t overflow_size = overflow_values.size();
  const std::size_t overflow_capacity = overflow_values.capacity();
  context->Expect(overflow_capacity > overflow_size,
                  "The release test should cover spare heap capacity.");

  HeapAllocation allocation;
  const std::size_t overflow_release_allocations =
      CountAllocations([&] { allocation = overflow_values.ReleaseHeap(); });
  context->ExpectEqual(overflow_release_allocations, std::size_t{0},
                       "Releasing heap storage should not allocate.");
  context->Expect(allocation.data() == overflow_data,
                  "Heap release should transfer the original allocation.");
  context->ExpectEqual(allocation.size(), overflow_size,
                       "Heap release should preserve the logical size.");
  context->ExpectEqual(allocation.capacity(), overflow_capacity,
                       "Heap release should preserve the allocation capacity.");
  ExpectHeapValues(context, allocation, {1, 2, 3, 4, 5},
                   "Heap release should preserve every value.");
  context->Expect(overflow_values.empty(),
                  "A heap release source should become empty.");
  context->ExpectEqual(overflow_values.capacity(), std::size_t{4},
                       "A heap release source should restore inline capacity.");

  HeapAllocation second_allocation = overflow_values.ReleaseHeap();
  context->Expect(second_allocation.empty(),
                  "Releasing the same source twice should return no heap.");
  context->Expect(overflow_values.empty(),
                  "A second heap release should leave the source empty.");
  overflow_values.assign({9, 8});
  ExpectValues(context, overflow_values, {9, 8},
               "A heap release source should remain reusable.");

  HeapAllocation moved_allocation;
  moved_allocation = std::move(allocation);
  context->Expect(allocation.empty(),
                  "Moving a heap owner should empty the source owner.");
  context->Expect(moved_allocation.data() == overflow_data,
                  "Moving a heap owner should preserve its allocation.");
  const std::size_t owner_deallocations = CountDeallocations(
      [&] { HeapAllocation final_allocation{std::move(moved_allocation)}; });
  context->ExpectEqual(
      owner_deallocations, std::size_t{1},
      "A moved heap owner should deallocate its allocation exactly once.");
  context->Expect(moved_allocation.empty(),
                  "Moving a heap owner should leave it non-owning.");

  Inline4 released_values{4, 3, 2, 1, 0};
  released_values.reserve(10);
  HeapAllocation released_allocation = released_values.ReleaseHeap();
  const std::size_t released_capacity = released_allocation.capacity();
  std::size_t* const released_data = released_allocation.release();
  context->Expect(released_allocation.empty(),
                  "Explicit release should empty the heap owner.");
  context->Expect(released_allocation.release() == nullptr,
                  "Explicit release should return the allocation only once.");
  context->ExpectEqual(released_data[0], std::size_t{4},
                       "Explicit release should return the owned values.");
  std::allocator<std::size_t> allocator;
  allocator.deallocate(released_data, released_capacity);
}

void TestCopySemantics(TestContext* context) {
  Inline4 inline_source{1, 2, 3};
  Inline4 inline_copy{inline_source};
  inline_source[0] = 9;
  ExpectValues(context, inline_copy, {1, 2, 3},
               "An inline copy should own independent values.");

  Inline4 overflow_source{1, 2, 3, 4, 5};
  Inline4 overflow_copy{overflow_source};
  overflow_source[0] = 9;
  ExpectValues(context, overflow_copy, {1, 2, 3, 4, 5},
               "An overflow copy should own independent values.");

  Inline4 copy_assignment_source{4, 5, 6, 7, 8};
  Inline4 copy_assigned{0};
  copy_assigned = copy_assignment_source;
  copy_assignment_source[1] = 0;
  ExpectValues(context, copy_assigned, {4, 5, 6, 7, 8},
               "Copy assignment should own independent values.");

  Inline4 inline_assignment_source{4, 5, 6};
  Inline4 heap_copy_assigned{0, 1, 2, 3, 4};
  heap_copy_assigned = inline_assignment_source;
  inline_assignment_source[0] = 0;
  ExpectValues(context, heap_copy_assigned, {4, 5, 6},
               "Copy assignment should replace heap values with an "
               "independent inline copy.");
  context->ExpectEqual(
      heap_copy_assigned.capacity(), std::size_t{4},
      "Copy assignment from inline values should restore inline storage.");

  Inline4 self_assigned{1, 2, 3};
  self_assigned = self_assigned;
  context->Expect(self_assigned == std::vector<std::size_t>({1, 2, 3}),
                  "Self-assignment should preserve values.");
}

void TestMoveSemantics(TestContext* context) {
  Inline4 inline_construct_source{1, 2, 3};
  Inline4 inline_constructed{std::move(inline_construct_source)};
  ExpectValues(context, inline_constructed, {1, 2, 3},
               "Moving inline values should preserve them in the destination.");
  inline_construct_source = Inline4{9};
  ExpectValues(context, inline_construct_source, {9},
               "An inline move source should remain assignable.");

  Inline4 overflow_construct_source{1, 2, 3, 4, 5};
  Inline4 overflow_constructed{std::move(overflow_construct_source)};
  ExpectValues(
      context, overflow_constructed, {1, 2, 3, 4, 5},
      "Moving overflow values should preserve them in the destination.");
  overflow_construct_source = Inline4{9};
  ExpectValues(context, overflow_construct_source, {9},
               "An overflow move source should remain assignable.");

  Inline4 inline_assignment_source{4, 5, 6};
  Inline4 inline_assigned{0, 1, 2, 3, 4};
  inline_assigned = std::move(inline_assignment_source);
  ExpectValues(context, inline_assigned, {4, 5, 6},
               "Move assignment should preserve inline values.");
  context->ExpectEqual(
      inline_assigned.capacity(), std::size_t{4},
      "Move assignment from inline values should restore inline storage.");
  inline_assignment_source = Inline4{9};
  ExpectValues(context, inline_assignment_source, {9},
               "An inline move-assignment source should remain assignable.");

  Inline4 overflow_assignment_source{4, 5, 6, 7, 8};
  Inline4 overflow_assigned{0};
  overflow_assigned = std::move(overflow_assignment_source);
  ExpectValues(context, overflow_assigned, {4, 5, 6, 7, 8},
               "Move assignment should preserve overflow values.");
  overflow_assignment_source = Inline4{9};
  ExpectValues(context, overflow_assignment_source, {9},
               "An overflow move-assignment source should remain assignable.");

  Inline4 self_moved{1, 2, 3};
  self_moved = std::move(self_moved);
  ExpectValues(context, self_moved, {1, 2, 3},
               "Self-move assignment should preserve values.");
}

}  // namespace

int main() {
  TestContext context;

  TestConstruction(&context);
  TestAccessorsAndIterators(&context);
  TestEquality(&context);
  TestMutation(&context);
  TestHeapRelease(&context);
  TestCopySemantics(&context);
  TestMoveSemantics(&context);

  return context.ExitCode();
}
