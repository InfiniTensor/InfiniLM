#include <array>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <sstream>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/shape_strides_storage.h"
#include "test_helper.h"

namespace {

using ShapeStridesStorage =
    infini::rt::detail::ShapeStridesStorage<std::size_t, std::ptrdiff_t, 4>;
using DefaultStridesTag = infini::rt::detail::DefaultStridesTag;
using Shape = ShapeStridesStorage::Shape;
using Strides = ShapeStridesStorage::Strides;
using infini::rt::test::TestContext;

static_assert(std::is_copy_constructible_v<ShapeStridesStorage>);
static_assert(std::is_nothrow_move_constructible_v<ShapeStridesStorage>);
static_assert(!std::is_copy_assignable_v<ShapeStridesStorage>);
static_assert(!std::is_move_assignable_v<ShapeStridesStorage>);

template <typename T, typename Expected>
void ExpectView(TestContext* context,
                infini::rt::detail::MetadataView<T> actual,
                std::initializer_list<Expected> expected,
                std::string_view message) {
  context->ExpectEqual(actual, std::vector<T>(expected.begin(), expected.end()),
                       message);
}

template <typename T>
void ExpectContiguous(TestContext* context,
                      infini::rt::detail::MetadataView<T> view,
                      std::string_view message) {
  context->Expect(view.data() == view.begin(), message);
  context->Expect(view.end() == view.data() + view.size(), message);

  for (std::size_t index = 0; index < view.size(); ++index) {
    context->Expect(&view[index] == view.data() + index, message);
  }
}

template <typename T>
class InputRange {
 public:
  explicit InputRange(std::istream* stream) : stream_(stream) {}

  std::istream_iterator<T> begin() const {
    return std::istream_iterator<T>{*stream_};
  }

  std::istream_iterator<T> end() const { return std::istream_iterator<T>{}; }

 private:
  std::istream* stream_;
};

void TestEmptyMetadata(TestContext* context) {
  const ShapeStridesStorage metadata;

  context->Expect(metadata.shape().empty(),
                  "Default tensor metadata should have an empty shape.");
  context->Expect(metadata.strides().empty(),
                  "Default tensor metadata should have empty strides.");
  context->ExpectEqual(metadata.shape().size(), std::size_t{0},
                       "Default shape size should be zero.");
  context->ExpectEqual(metadata.strides().size(), std::size_t{0},
                       "Default strides size should be zero.");

  const Shape shape;
  const Strides strides;
  const ShapeStridesStorage explicit_empty{shape, strides};
  context->Expect(explicit_empty.shape().empty(),
                  "Explicit rank-zero metadata should have an empty shape.");
  context->Expect(explicit_empty.strides().empty(),
                  "Explicit rank-zero metadata should have empty strides.");

  const std::array<std::size_t, 0> empty_shape{};
  const std::array<std::ptrdiff_t, 0> empty_strides{};
  const ShapeStridesStorage empty_array_metadata{empty_shape, empty_strides};
  context->Expect(empty_array_metadata.shape().empty() &&
                      empty_array_metadata.strides().empty(),
                  "Empty standard arrays should construct rank-zero metadata.");
}

void TestInlineMetadata(TestContext* context) {
  const Shape shape{2, 3, 4, 5};
  const Strides strides{60, 20, 5, 1};
  const ShapeStridesStorage metadata{shape, strides};

  ExpectView(context, metadata.shape(), {2, 3, 4, 5},
             "Rank-four inline metadata should preserve shape values.");
  ExpectView(context, metadata.strides(), {60, 20, 5, 1},
             "Rank-four inline metadata should preserve stride values.");
  ExpectContiguous(context, metadata.shape(),
                   "Inline shape values should be contiguous.");
  ExpectContiguous(context, metadata.strides(),
                   "Inline stride values should be contiguous.");
  context->ExpectEqual(metadata.shape_size(), std::size_t{4},
                       "Direct inline shape size should match the view.");
  context->ExpectEqual(metadata.strides_size(), std::size_t{4},
                       "Direct inline strides size should match the view.");
  context->Expect(metadata.shape_data() == metadata.shape().data(),
                  "Direct inline shape data should match the view.");
  context->Expect(metadata.strides_data() == metadata.strides().data(),
                  "Direct inline strides data should match the view.");
}

void TestCombinedMetadata(TestContext* context) {
  const Shape shape{2, 3, 4, 5, 6};
  const Strides strides{360, 120, 30, 6, 1};
  const ShapeStridesStorage exact_lvalue{shape, strides};

  ExpectView(context, exact_lvalue.shape(), {2, 3, 4, 5, 6},
             "Rank-five lvalue metadata should preserve shape values.");
  ExpectView(context, exact_lvalue.strides(), {360, 120, 30, 6, 1},
             "Rank-five lvalue metadata should preserve stride values.");
  ExpectContiguous(context, exact_lvalue.shape(),
                   "Combined shape values should be contiguous.");
  ExpectContiguous(context, exact_lvalue.strides(),
                   "Combined stride values should be contiguous.");
  context->ExpectEqual(exact_lvalue.shape_size(), std::size_t{5},
                       "Direct heap shape size should match the view.");
  context->ExpectEqual(exact_lvalue.strides_size(), std::size_t{5},
                       "Direct heap strides size should match the view.");
  context->Expect(exact_lvalue.shape_data() == exact_lvalue.shape().data(),
                  "Direct heap shape data should match the view.");
  context->Expect(exact_lvalue.strides_data() == exact_lvalue.strides().data(),
                  "Direct heap strides data should match the view.");

  const std::array<unsigned int, 5> generic_shape{7, 8, 9, 10, 11};
  const std::array<int, 5> generic_strides{7920, 990, 110, 11, 1};
  const ShapeStridesStorage generic{generic_shape, generic_strides};
  ExpectView(context, generic.shape(), {7, 8, 9, 10, 11},
             "Generic rank-five metadata should convert shape values.");
  ExpectView(context, generic.strides(), {7920, 990, 110, 11, 1},
             "Generic rank-five metadata should convert stride values.");
}

void TestSplitRvalueMetadata(TestContext* context) {
  const ShapeStridesStorage temporary_values{Shape{2, 3, 4, 5, 6},
                                             Strides{360, 120, 30, 6, 1}};
  ExpectView(context, temporary_values.shape(), {2, 3, 4, 5, 6},
             "Exact rvalue metadata should preserve shape values.");
  ExpectView(context, temporary_values.strides(), {360, 120, 30, 6, 1},
             "Exact rvalue metadata should preserve stride values.");

  Shape shape{3, 4, 5, 6, 7};
  Strides strides{840, 210, 42, 7, 1};
  const ShapeStridesStorage pre_moved{std::move(shape), std::move(strides)};
  ExpectView(context, pre_moved.shape(), {3, 4, 5, 6, 7},
             "Pre-moved metadata should preserve shape values.");
  ExpectView(context, pre_moved.strides(), {840, 210, 42, 7, 1},
             "Pre-moved metadata should preserve stride values.");
  ExpectContiguous(context, pre_moved.shape(),
                   "Split shape values should be contiguous.");
  ExpectContiguous(context, pre_moved.strides(),
                   "Split stride values should be contiguous.");
}

ShapeStridesStorage CopyPastSourceLifetime(TestContext* context) {
  const ShapeStridesStorage source{Shape{2, 3, 4, 5, 6},
                                   Strides{360, 120, 30, 6, 1}};
  ShapeStridesStorage copy{source};

  context->Expect(copy.shape().data() != source.shape().data(),
                  "A metadata copy should own separate shape storage.");
  context->Expect(copy.strides().data() != source.strides().data(),
                  "A metadata copy should own separate stride storage.");

  return copy;
}

ShapeStridesStorage MovePastSourceLifetime() {
  ShapeStridesStorage source{Shape{3, 4, 5, 6, 7}, Strides{840, 210, 42, 7, 1}};
  ShapeStridesStorage moved{std::move(source)};

  return moved;
}

void TestCopyAndMoveOwnership(TestContext* context) {
  const ShapeStridesStorage copy = CopyPastSourceLifetime(context);
  ExpectView(context, copy.shape(), {2, 3, 4, 5, 6},
             "A copy should remain valid after its source is destroyed.");
  ExpectView(context, copy.strides(), {360, 120, 30, 6, 1},
             "Copied strides should survive source destruction.");

  const ShapeStridesStorage moved = MovePastSourceLifetime();
  ExpectView(context, moved.shape(), {3, 4, 5, 6, 7},
             "Moved metadata should survive source destruction.");
  ExpectView(context, moved.strides(), {840, 210, 42, 7, 1},
             "Moved strides should survive source destruction.");
}

void TestMixedOwnership(TestContext* context) {
  Shape moved_shape{2, 3, 4, 5, 6};
  const Strides borrowed_strides{360, 120, 30, 6, 1};
  const ShapeStridesStorage shape_rvalue{std::move(moved_shape),
                                         borrowed_strides};
  ExpectView(context, shape_rvalue.shape(), {2, 3, 4, 5, 6},
             "A moved shape with lvalue strides should preserve shape.");
  ExpectView(context, shape_rvalue.strides(), {360, 120, 30, 6, 1},
             "A moved shape with lvalue strides should preserve strides.");

  const Shape borrowed_shape{3, 4, 5, 6, 7};
  Strides moved_strides{840, 210, 42, 7, 1};
  const ShapeStridesStorage strides_rvalue{borrowed_shape,
                                           std::move(moved_strides)};
  ExpectView(context, strides_rvalue.shape(), {3, 4, 5, 6, 7},
             "An lvalue shape with moved strides should preserve shape.");
  ExpectView(context, strides_rvalue.strides(), {840, 210, 42, 7, 1},
             "An lvalue shape with moved strides should preserve strides.");
}

void TestDefaultStrides(TestContext* context) {
  const ShapeStridesStorage inline_metadata{Shape{2, 3, 4, 5},
                                            DefaultStridesTag{}};
  ExpectView(context, inline_metadata.strides(), {60, 20, 5, 1},
             "Default inline strides should be row-major.");

  Shape shape{2, 3, 4, 5, 6};
  const ShapeStridesStorage heap_metadata{std::move(shape),
                                          DefaultStridesTag{}};
  ExpectView(context, heap_metadata.shape(), {2, 3, 4, 5, 6},
             "Default-stride construction should preserve shape.");
  ExpectView(context, heap_metadata.strides(), {360, 120, 30, 6, 1},
             "Default rank-five strides should be row-major.");
}

void TestIndependentViewLengths(TestContext* context) {
  const ShapeStridesStorage longer_shape{Shape{2, 3, 4, 5, 6},
                                         Strides{20, 5, 1}};
  context->ExpectEqual(longer_shape.shape().size(), std::size_t{5},
                       "Shape length should be preserved independently.");
  context->ExpectEqual(longer_shape.strides().size(), std::size_t{3},
                       "Stride length should be preserved independently.");
  context->ExpectEqual(longer_shape.shape_size(), std::size_t{5},
                       "Direct shape size should remain independent.");
  context->ExpectEqual(longer_shape.strides_size(), std::size_t{3},
                       "Direct strides size should remain independent.");
  ExpectView(context, longer_shape.shape(), {2, 3, 4, 5, 6},
             "A longer shape should preserve all shape values.");
  ExpectView(context, longer_shape.strides(), {20, 5, 1},
             "A shorter stride range should preserve all stride values.");

  const ShapeStridesStorage longer_strides{Shape{2, 3, 4},
                                           Strides{360, 120, 30, 6, 1}};
  context->ExpectEqual(longer_strides.shape().size(), std::size_t{3},
                       "Shorter shape length should be preserved.");
  context->ExpectEqual(longer_strides.strides().size(), std::size_t{5},
                       "Longer stride length should be preserved.");
  context->ExpectEqual(longer_strides.shape_size(), std::size_t{3},
                       "Direct shorter shape size should be preserved.");
  context->ExpectEqual(longer_strides.strides_size(), std::size_t{5},
                       "Direct longer strides size should be preserved.");
}

void TestInputRanges(TestContext* context) {
  std::istringstream shape_stream{"2 3 4 5 6"};
  std::istringstream strides_stream{"360 120 30 6 1"};
  const InputRange<std::size_t> shape{&shape_stream};
  const InputRange<std::ptrdiff_t> strides{&strides_stream};
  const ShapeStridesStorage metadata{shape, strides};

  ExpectView(context, metadata.shape(), {2, 3, 4, 5, 6},
             "Input ranges should be consumed once for shape values.");
  ExpectView(context, metadata.strides(), {360, 120, 30, 6, 1},
             "Input ranges should be consumed once for stride values.");
}

}  // namespace

int main() {
  TestContext context;

  TestEmptyMetadata(&context);
  TestInlineMetadata(&context);
  TestCombinedMetadata(&context);
  TestSplitRvalueMetadata(&context);
  TestCopyAndMoveOwnership(&context);
  TestMixedOwnership(&context);
  TestDefaultStrides(&context);
  TestIndependentViewLengths(&context);
  TestInputRanges(&context);

  return context.ExitCode();
}
