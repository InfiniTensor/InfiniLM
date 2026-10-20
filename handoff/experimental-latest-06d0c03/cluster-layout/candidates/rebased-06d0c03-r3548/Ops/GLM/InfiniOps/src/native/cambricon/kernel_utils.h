#ifndef INFINI_OPS_CAMBRICON_KERNEL_UTILS_H_
#define INFINI_OPS_CAMBRICON_KERNEL_UTILS_H_

#include <cstddef>

#ifdef __BANG__

namespace infini::ops::cambricon::kernel_utils {

struct TaskRange {
  std::size_t begin;
  std::size_t end;
};

__mlu_device__ inline TaskRange GetTaskRange(std::size_t size) {
  const std::size_t elements_per_task = (size + taskDim - 1) / taskDim;
  const std::size_t begin = taskId * elements_per_task;
  const std::size_t end =
      begin + elements_per_task < size ? begin + elements_per_task : size;
  return {begin, end};
}

__mlu_device__ inline ptrdiff_t LogicalToOffset(std::size_t logical_index,
                                                int ndim,
                                                const std::size_t* shape,
                                                const ptrdiff_t* strides) {
  ptrdiff_t offset = 0;
  for (int dim = ndim - 1; dim >= 0; --dim) {
    const std::size_t coordinate = logical_index % shape[dim];
    logical_index /= shape[dim];
    offset += static_cast<ptrdiff_t>(coordinate) * strides[dim];
  }
  return offset;
}

}  // namespace infini::ops::cambricon::kernel_utils

#endif  // __BANG__

#endif  // INFINI_OPS_CAMBRICON_KERNEL_UTILS_H_
