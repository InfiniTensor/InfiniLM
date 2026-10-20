#ifndef INFINI_OPS_CAMBRICON_CNNL_UTILS_H_
#define INFINI_OPS_CAMBRICON_CNNL_UTILS_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include "native/cambricon/common.h"
#include "tensor.h"

#define INFINI_OPS_CNNL_CHECK(call)                                      \
  do {                                                                   \
    const auto cnnl_status = (call);                                     \
    assert(cnnl_status == CNNL_STATUS_SUCCESS && "`" #call "` failed."); \
    (void)cnnl_status;                                                   \
  } while (false)

namespace infini::ops::cnnl_utils {

struct HandleDeleter {
  using pointer = cnnlHandle_t;

  void operator()(pointer handle) const noexcept {
    if (handle) {
      (void)cnnlDestroy(handle);
    }
  }
};

using Handle =
    std::unique_ptr<std::remove_pointer_t<cnnlHandle_t>, HandleDeleter>;

inline Handle CreateHandle() {
  cnnlHandle_t handle{nullptr};
  INFINI_OPS_CNNL_CHECK(cnnlCreate(&handle));

  return Handle{handle};
}

struct TensorDescriptorDeleter {
  using pointer = cnnlTensorDescriptor_t;

  void operator()(pointer desc) const noexcept {
    if (desc) {
      (void)cnnlDestroyTensorDescriptor(desc);
    }
  }
};

using TensorDescriptor =
    std::unique_ptr<std::remove_pointer_t<cnnlTensorDescriptor_t>,
                    TensorDescriptorDeleter>;

inline TensorDescriptor CreateTensorDescriptor() {
  cnnlTensorDescriptor_t desc{nullptr};
  INFINI_OPS_CNNL_CHECK(cnnlCreateTensorDescriptor(&desc));

  return TensorDescriptor{desc};
}

inline void SetTensorDescriptor(cnnlTensorDescriptor_t desc, DataType dtype,
                                const Tensor::Shape& shape,
                                const Tensor::Strides& strides) {
  assert(shape.size() == strides.size() &&
         "`CNNL tensor descriptor` requires matching shape and strides.");

  const auto cnnl_dtype = GetDataType(dtype);
  assert(cnnl_dtype != CNNL_DTYPE_INVALID &&
         "`CNNL tensor descriptor` does not support this data type.");

  const auto cnnl_shape =
      shape.empty() ? std::vector<std::int64_t>{1}
                    : std::vector<std::int64_t>(shape.begin(), shape.end());
  const auto cnnl_strides =
      strides.empty()
          ? std::vector<std::int64_t>{1}
          : std::vector<std::int64_t>(strides.begin(), strides.end());
  const auto ndim = static_cast<int>(cnnl_shape.size());

  INFINI_OPS_CNNL_CHECK(
      cnnlSetTensorDescriptorEx_v2(desc, CNNL_LAYOUT_ARRAY, cnnl_dtype, ndim,
                                   cnnl_shape.data(), cnnl_strides.data()));
}

inline TensorDescriptor MakeTensorDescriptor(DataType dtype,
                                             const Tensor::Shape& shape,
                                             const Tensor::Strides& strides) {
  auto desc = CreateTensorDescriptor();
  SetTensorDescriptor(desc.get(), dtype, shape, strides);

  return desc;
}

}  // namespace infini::ops::cnnl_utils

#endif
