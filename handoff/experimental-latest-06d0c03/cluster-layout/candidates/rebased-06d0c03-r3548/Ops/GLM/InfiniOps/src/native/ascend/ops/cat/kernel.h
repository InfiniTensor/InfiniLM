#ifndef INFINI_OPS_ASCEND_CAT_KERNEL_H_
#define INFINI_OPS_ASCEND_CAT_KERNEL_H_

#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_cat.h"
#include "base/cat.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kAscend> : public Cat {
 public:
  Operator(const std::vector<Tensor> tensors, const int64_t dim, Tensor out)
      : Cat(tensors, dim, out), out_cache_(out) {
    in_caches_.reserve(input_count_);
    for (const auto& tensor : tensors) {
      in_caches_.emplace_back(tensor);
    }
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    // Null cached descriptors — see `AclTensorCache::release()`.  The input
    // descriptors are referenced by the Repeatable `executor_` via
    // `tensor_list_`, so every `in_caches_[i]` must be released alongside
    // `out_cache_`; otherwise `~AclTensorCache()` double-frees them when the
    // vector destructs.
    for (auto& c : in_caches_) {
      c.release();
    }
    out_cache_.release();

    if (tensor_list_) aclDestroyTensorList(tensor_list_);
  }

  void operator()(const std::vector<Tensor> tensors, const int64_t /*dim*/,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_out = out_cache_.get(out.data());

    if (!executor_) {
      // First call: create descriptors, tensor list, and executor.
      std::vector<aclTensor*> acl_tensors(input_count_);
      for (size_t i = 0; i < input_count_; ++i) {
        acl_tensors[i] =
            in_caches_[i].get(const_cast<void*>(tensors[i].data()));
      }

      tensor_list_ =
          aclCreateTensorList(const_cast<const aclTensor**>(acl_tensors.data()),
                              static_cast<uint64_t>(input_count_));

      aclnnCatGetWorkspaceSize(tensor_list_, dim_, t_out, &ws_size_,
                               &executor_);
      aclSetAclOpExecutorRepeatable(executor_);
    } else {
      // Subsequent calls: update data pointers on cached descriptors via
      // `aclSetRawTensorAddr`.  The executor holds references to the same
      // `aclTensor*` objects inside `tensor_list_`, so updating their data
      // pointers is sufficient — no `aclSetInputTensorAddr` needed.
      for (size_t i = 0; i < input_count_; ++i) {
        in_caches_[i].get(const_cast<void*>(tensors[i].data()));
      }
      aclSetOutputTensorAddr(executor_, 0, t_out, out.data());
    }

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, ws_size_);
    aclnnCat(arena.buf, ws_size_, executor_, stream);
  }

 private:
  mutable std::vector<ascend::AclTensorCache> in_caches_;

  mutable ascend::AclTensorCache out_cache_;

  mutable aclTensorList* tensor_list_ = nullptr;

  mutable aclOpExecutor* executor_ = nullptr;

  mutable uint64_t ws_size_ = 0;
};

}  // namespace infini::ops

#endif
