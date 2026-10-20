#ifndef INFINI_OPS_ASCEND_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_ASCEND_CAUSAL_SOFTMAX_KERNEL_H_

#include <limits>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_copy.h"
#include "aclnn_masked_fill_scalar.h"
#include "aclnn_softmax.h"
#include "base/causal_softmax.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

// CANN 8.5 has no single API covering causal-mask-then-softmax. Decompose the
// operation into a stride-aware copy, masked fill, and last-dimension softmax.
template <>
class Operator<CausalSoftmax, Device::Type::kAscend> : public CausalSoftmax {
 public:
  Operator(const Tensor input, Tensor out)
      : CausalSoftmax(input, out), in_cache_(input), out_cache_(out) {
    temp_size_ = input.numel() * kDataTypeToSize.at(dtype_);
    Tensor temp_tensor{nullptr, input.shape(), input.dtype(), input.device()};
    temp_cache_ = ascend::AclTensorCache(temp_tensor);

    // `mask[i][j] = 1` when key position `j` is not visible to query `i`.
    // Shape `(seq_len, total_seq_len)` broadcasts over leading dimensions.
    size_t mask_elems = seq_len_ * total_seq_len_;
    std::vector<uint8_t> mask_host(mask_elems, 0);
    for (size_t i = 0; i < seq_len_; ++i) {
      auto vis_end = static_cast<int64_t>(total_seq_len_ - seq_len_ + i);
      for (auto j = vis_end + 1; j < static_cast<int64_t>(total_seq_len_);
           ++j) {
        mask_host[i * total_seq_len_ + j] = 1;
      }
    }

    aclrtMalloc(&mask_buf_, mask_elems, ACL_MEM_MALLOC_NORMAL_ONLY);
    aclrtMemcpy(mask_buf_, mask_elems, mask_host.data(), mask_elems,
                ACL_MEMCPY_HOST_TO_DEVICE);

    std::vector<int64_t> mshape = {static_cast<int64_t>(seq_len_),
                                   static_cast<int64_t>(total_seq_len_)};
    std::vector<int64_t> mstrides = {static_cast<int64_t>(total_seq_len_), 1};
    mask_tensor_ = aclCreateTensor(mshape.data(), mshape.size(), ACL_BOOL,
                                   mstrides.data(), 0, ACL_FORMAT_ND,
                                   mshape.data(), mshape.size(), mask_buf_);

    // `aclCreateScalar` stores the pointer, so the backing value is a member.
    neg_inf_ = aclCreateScalar(&neg_inf_storage_, ACL_FLOAT);
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    if (mask_tensor_) aclDestroyTensor(mask_tensor_);
    if (mask_buf_) aclrtFree(mask_buf_);
    if (neg_inf_) aclDestroyScalar(neg_inf_);
  }

  void operator()(const Tensor input, Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto& temp = ascend::GetWorkspacePool().Ensure(stream, temp_size_, "temp");

    // Descriptors stay owned by the cached operator and outlive all queued
    // work; only their raw addresses change between invocations.
    auto t_in = in_cache_.get(const_cast<void*>(input.data()));
    auto t_out = out_cache_.get(out.data());
    auto t_temp = temp_cache_.get(temp.buf);

    // CANN consumes these executors even when they are marked repeatable.
    // Acquire a fresh executor for every transformer layer invocation.
    aclOpExecutor* copy_exec = nullptr;
    uint64_t copy_ws = 0;
    aclnnInplaceCopyGetWorkspaceSize(t_temp, t_in, &copy_ws, &copy_exec);
    auto& copy_arena = ascend::GetWorkspacePool().Ensure(stream, copy_ws);
    aclnnInplaceCopy(copy_arena.buf, copy_ws, copy_exec, stream);

    aclOpExecutor* fill_exec = nullptr;
    uint64_t fill_ws = 0;
    aclnnInplaceMaskedFillScalarGetWorkspaceSize(t_temp, mask_tensor_, neg_inf_,
                                                 &fill_ws, &fill_exec);
    auto& fill_arena = ascend::GetWorkspacePool().Ensure(stream, fill_ws);
    aclnnInplaceMaskedFillScalar(fill_arena.buf, fill_ws, fill_exec, stream);

    constexpr int64_t kLastDim = -1;
    aclOpExecutor* softmax_exec = nullptr;
    uint64_t softmax_ws = 0;
    aclnnSoftmaxGetWorkspaceSize(t_temp, kLastDim, t_out, &softmax_ws,
                                 &softmax_exec);
    auto& softmax_arena = ascend::GetWorkspacePool().Ensure(stream, softmax_ws);
    aclnnSoftmax(softmax_arena.buf, softmax_ws, softmax_exec, stream);
  }

 private:
  mutable ascend::AclTensorCache in_cache_;

  mutable ascend::AclTensorCache out_cache_;

  mutable ascend::AclTensorCache temp_cache_;

  float neg_inf_storage_ = -std::numeric_limits<float>::infinity();

  uint64_t temp_size_ = 0;

  void* mask_buf_ = nullptr;

  aclTensor* mask_tensor_ = nullptr;

  aclScalar* neg_inf_ = nullptr;
};

}  // namespace infini::ops

#endif
