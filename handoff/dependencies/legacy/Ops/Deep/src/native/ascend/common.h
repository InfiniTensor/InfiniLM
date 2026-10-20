#ifndef INFINI_OPS_ASCEND_COMMON_H_
#define INFINI_OPS_ASCEND_COMMON_H_

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "native/ascend/data_type_.h"
#include "tensor.h"

namespace infini::ops::ascend {

// Check whether the ACL runtime is still usable.
//
// During process shutdown the CANN runtime may be torn down before C++
// static destructors run.  Calling `aclrtGetDevice` is the cheapest
// probe — it fails once the runtime is gone.  Destructors that call
// ACL/ATB APIs must guard with this to avoid use-after-finalize crashes.
inline bool IsAclRuntimeAlive() {
  int32_t dev_id = -1;

  return aclrtGetDevice(&dev_id) == ACL_SUCCESS;
}

// Build an `aclTensor` descriptor from an InfiniOps `Tensor`.
//
// When `transpose_last2` is true the last two dimensions are swapped in the
// descriptor (shape and strides) without copying data.  This is used by `Gemm`
// and `MatMul` to express a transpose via the view.
inline aclTensor* BuildAclTensor(const Tensor& t,
                                 bool transpose_last2 = false) {
  std::vector<int64_t> shape(t.shape().begin(), t.shape().end());
  std::vector<int64_t> strides(t.strides().begin(), t.strides().end());

  if (transpose_last2 && shape.size() >= 2) {
    auto n = shape.size();
    std::swap(shape[n - 2], shape[n - 1]);
    std::swap(strides[n - 2], strides[n - 1]);
  }

  // Compute the minimum physical storage needed for this strided view.
  // For contiguous tensors this equals `numel()`; for non-contiguous (gapped)
  // tensors it may be larger; for broadcast (stride-0) tensors it may be
  // smaller.  Passing the view shape as the storage shape causes
  // "ViewShape overlap" errors in ACLNN for non-contiguous inputs.
  int64_t storage_elems = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      storage_elems = 0;
      break;
    }
    if (strides[i] > 0 && shape[i] > 1) {
      storage_elems += static_cast<int64_t>(shape[i] - 1) * strides[i];
    }
  }
  std::vector<int64_t> storage_shape = {storage_elems};

  return aclCreateTensor(
      shape.data(), static_cast<int64_t>(shape.size()), ToAclDtype(t.dtype()),
      strides.data(),
      /*storageOffset=*/0, ACL_FORMAT_ND, storage_shape.data(),
      static_cast<int64_t>(storage_shape.size()), const_cast<void*>(t.data()));
}

// Pre-computed tensor metadata for descriptor reuse.
//
// Stores `shape`, `strides`, `storage_shape`, and `dtype` once (avoiding
// per-call heap allocations).  The `aclTensor` descriptor is created on the
// first `get()` call and its data pointer is updated in-place via
// `aclSetRawTensorAddr` on subsequent calls.
class AclTensorCache {
 public:
  AclTensorCache() = default;

  // Construct from explicit metadata (for device buffers not wrapped in
  // Tensor). Computes contiguous strides from shape.
  AclTensorCache(std::vector<int64_t> shape, aclDataType dtype, void* data)
      : shape_(std::move(shape)), dtype_(dtype) {
    strides_.resize(shape_.size());
    int64_t stride = 1;
    for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
      strides_[i] = stride;
      stride *= shape_[i];
    }
    storage_shape_ = {stride};

    if (data) {
      tensor_ = aclCreateTensor(
          shape_.data(), static_cast<int64_t>(shape_.size()), dtype_,
          strides_.data(),
          /*storageOffset=*/0, ACL_FORMAT_ND, storage_shape_.data(),
          static_cast<int64_t>(storage_shape_.size()), data);
    }
  }

  explicit AclTensorCache(const Tensor& t, bool transpose_last2 = false)
      : dtype_{ToAclDtype(t.dtype())} {
    shape_.assign(t.shape().begin(), t.shape().end());
    strides_.assign(t.strides().begin(), t.strides().end());

    if (transpose_last2 && shape_.size() >= 2) {
      auto n = shape_.size();
      std::swap(shape_[n - 2], shape_[n - 1]);
      std::swap(strides_[n - 2], strides_[n - 1]);
    }

    int64_t storage_elems = 1;
    for (size_t i = 0; i < shape_.size(); ++i) {
      if (shape_[i] == 0) {
        storage_elems = 0;
        break;
      }
      if (strides_[i] > 0 && shape_[i] > 1) {
        storage_elems += static_cast<int64_t>(shape_[i] - 1) * strides_[i];
      }
    }
    storage_shape_ = {storage_elems};
  }

  ~AclTensorCache() {
    if (tensor_ && IsAclRuntimeAlive()) {
      aclDestroyTensor(tensor_);
    }
  }

  AclTensorCache(const AclTensorCache&) = delete;

  AclTensorCache& operator=(const AclTensorCache&) = delete;

  AclTensorCache(AclTensorCache&& o) noexcept
      : shape_(std::move(o.shape_)),
        strides_(std::move(o.strides_)),
        storage_shape_(std::move(o.storage_shape_)),
        dtype_(o.dtype_),
        tensor_(o.tensor_) {
    o.tensor_ = nullptr;
  }

  AclTensorCache& operator=(AclTensorCache&& o) noexcept {
    if (this != &o) {
      if (tensor_) {
        aclDestroyTensor(tensor_);
      }
      shape_ = std::move(o.shape_);
      strides_ = std::move(o.strides_);
      storage_shape_ = std::move(o.storage_shape_);
      dtype_ = o.dtype_;
      tensor_ = o.tensor_;
      o.tensor_ = nullptr;
    }

    return *this;
  }

  // Null the cached descriptor pointer without calling `aclDestroyTensor`.
  // Call from the owning operator's destructor: the descriptor is still
  // referenced by a Repeatable `aclOpExecutor` which would be destroyed
  // alongside the tensor, and per CANN 8.5 docs that destruction is our
  // responsibility.  In practice `aclDestroyAclOpExecutor` during process
  // shutdown crashes even with `IsAclRuntimeAlive()` true — see `64c367c` —
  // so operators leak the executor at shutdown; skipping `aclDestroyTensor`
  // here keeps `~AclTensorCache` from double-freeing a descriptor the
  // executor still holds.
  void release() { tensor_ = nullptr; }

  // Explicitly destroy the cached tensor and clear the pointer.
  // Use only when the descriptor is owned outside any executor (e.g. an
  // intermediate tensor not passed to `aclnn*GetWorkspaceSize`).
  void destroy() {
    if (tensor_) {
      aclDestroyTensor(tensor_);
      tensor_ = nullptr;
    }
  }

  // Update the data pointer and return the cached descriptor.
  aclTensor* get(void* data) const {
    if (tensor_) {
      aclSetRawTensorAddr(tensor_, data);

      return tensor_;
    }

    tensor_ = aclCreateTensor(
        shape_.data(), static_cast<int64_t>(shape_.size()), dtype_,
        strides_.data(),
        /*storageOffset=*/0, ACL_FORMAT_ND, storage_shape_.data(),
        static_cast<int64_t>(storage_shape_.size()), data);

    return tensor_;
  }

 private:
  std::vector<int64_t> shape_;

  std::vector<int64_t> strides_;

  std::vector<int64_t> storage_shape_;

  aclDataType dtype_{ACL_DT_UNDEFINED};

  mutable aclTensor* tensor_ = nullptr;
};

}  // namespace infini::ops::ascend

#endif
