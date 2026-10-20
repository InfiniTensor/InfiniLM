#ifndef INFINI_OPS_ASCEND_ATB_COMMON__H_
#define INFINI_OPS_ASCEND_ATB_COMMON__H_

#ifdef INFINI_HAS_ATB

#include <cassert>
#include <cstdint>
#include <mutex>
#include <vector>

#include "acl/acl.h"
#include "atb/context.h"
#include "atb/operation.h"
#include "atb/types.h"
#include "native/ascend/data_type_.h"
#include "tensor.h"

namespace infini::ops::ascend {

// Thread-local ATB context.
//
// ATB requires a `Context` for Setup/Execute.  Creating one per call is
// expensive (internal tiling buffer allocation), so we cache one per thread.
// `SetExecuteStream` is called before every `Execute` to match the caller's
// stream.
inline atb::Context*& ThreadLocalAtbContext() {
  thread_local atb::Context* ctx = nullptr;

  return ctx;
}

inline atb::Context* GetAtbContext(aclrtStream stream) {
  auto*& ctx = ThreadLocalAtbContext();

  if (!ctx) {
    atb::Status s = atb::CreateContext(&ctx);
    assert(s == atb::NO_ERROR && "atb::CreateContext failed");
  }

  atb::Status s = ctx->SetExecuteStream(stream);
  assert(s == atb::NO_ERROR && "atb::Context::SetExecuteStream failed");

  return ctx;
}

// Build an `atb::Tensor` from an InfiniOps Tensor.
//
// Sets dtype, ND format, shape dimensions, and the device data pointer.
// The caller must keep the InfiniOps Tensor alive for the duration of the
// ATB operation.
inline atb::Tensor ToAtbTensor(const Tensor& t) {
  atb::Tensor out;
  out.desc.dtype = ToAclDtype(t.dtype());
  out.desc.format = ACL_FORMAT_ND;
  out.desc.shape.dimNum = t.ndim();
  assert(t.ndim() <= atb::MAX_DIM);

  for (uint64_t i = 0; i < t.ndim(); ++i) {
    out.desc.shape.dims[i] = static_cast<int64_t>(t.size(i));
  }

  out.deviceData = const_cast<void*>(t.data());
  out.dataSize = static_cast<uint64_t>(t.numel()) * t.element_size();

  return out;
}

// Build an `atb::Tensor` from explicit shape, dtype, and data pointer.
//
// Useful for sub-views of a larger buffer (e.g. K-cache and V-cache halves
// of a fused KV cache tensor).
inline atb::Tensor ToAtbTensor(const std::vector<int64_t>& shape,
                               aclDataType dtype, void* data,
                               uint64_t data_size) {
  atb::Tensor out;
  out.desc.dtype = dtype;
  out.desc.format = ACL_FORMAT_ND;
  out.desc.shape.dimNum = shape.size();
  assert(shape.size() <= atb::MAX_DIM);

  for (size_t i = 0; i < shape.size(); ++i) {
    out.desc.shape.dims[i] = shape[i];
  }

  out.deviceData = data;
  out.dataSize = data_size;

  return out;
}

}  // namespace infini::ops::ascend

#endif  // INFINI_HAS_ATB

#endif  // INFINI_OPS_ASCEND_ATB_COMMON__H_
