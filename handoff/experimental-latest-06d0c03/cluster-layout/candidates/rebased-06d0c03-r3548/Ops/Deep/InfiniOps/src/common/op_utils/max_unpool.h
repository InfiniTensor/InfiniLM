#ifndef INFINI_OPS_COMMON_OP_UTILS_MAX_UNPOOL_H_
#define INFINI_OPS_COMMON_OP_UTILS_MAX_UNPOOL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "operator.h"

namespace infini::ops::max_unpool_detail {

template <std::size_t SpatialDimensions>
inline std::pair<std::vector<int64_t>, std::vector<int64_t>> ResolveGeometry(
    const Tensor input, const std::vector<int64_t>& kernel_size,
    const std::optional<std::vector<int64_t>>& stride,
    const std::vector<int64_t>& padding,
    const std::optional<std::vector<int64_t>>& output_size) {
  assert((input.ndim() == SpatialDimensions + 1 ||
          input.ndim() == SpatialDimensions + 2) &&
         "`MaxUnpool` input rank must include the spatial dimensions");
  assert(kernel_size.size() == SpatialDimensions &&
         "`MaxUnpool` `kernel_size` has the wrong length");
  assert(padding.size() == SpatialDimensions &&
         "`MaxUnpool` `padding` has the wrong length");

  auto resolved_stride = stride.value_or(kernel_size);
  assert(resolved_stride.size() == SpatialDimensions &&
         "`MaxUnpool` `stride` has the wrong length");

  std::vector<int64_t> default_size(SpatialDimensions);

  for (std::size_t dim = 0; dim < SpatialDimensions; ++dim) {
    assert(kernel_size[dim] > 0 &&
           "`MaxUnpool` requires positive `kernel_size` values");
    assert(resolved_stride[dim] > 0 &&
           "`MaxUnpool` requires positive `stride` values");
    assert(padding[dim] >= 0 &&
           "`MaxUnpool` requires non-negative `padding` values");

    const auto input_size = static_cast<int64_t>(
        input.size(input.ndim() - SpatialDimensions + dim));
    default_size[dim] = (input_size - 1) * resolved_stride[dim] +
                        kernel_size[dim] - 2 * padding[dim];
  }

  auto resolved_output_size = output_size.value_or(default_size);

  if (output_size.has_value()) {
    if (resolved_output_size.size() == SpatialDimensions + 2) {
      resolved_output_size.erase(resolved_output_size.begin(),
                                 resolved_output_size.begin() + 2);
    }

    assert(resolved_output_size.size() == SpatialDimensions &&
           "`MaxUnpool` `output_size` has the wrong length");

    for (std::size_t dim = 0; dim < SpatialDimensions; ++dim) {
      const auto min_size = default_size[dim] - resolved_stride[dim];
      const auto max_size = default_size[dim] + resolved_stride[dim];
      assert((min_size < resolved_output_size[dim] &&
              resolved_output_size[dim] < max_size) &&
             "`MaxUnpool` `output_size` is outside the valid range");
    }
  }

  for (const auto value : resolved_output_size) {
    assert(value >= 0 && "`MaxUnpool` requires non-negative output dimensions");
  }

  return {std::move(resolved_output_size), std::move(resolved_stride)};
}

}  // namespace infini::ops::max_unpool_detail

#endif
