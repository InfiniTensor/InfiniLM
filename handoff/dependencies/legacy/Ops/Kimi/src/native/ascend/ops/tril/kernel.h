#ifndef INFINI_OPS_ASCEND_TRIL_KERNEL_H_
#define INFINI_OPS_ASCEND_TRIL_KERNEL_H_

#include <cstdio>
#include <cstdlib>

#include "aclnnop/aclnn_tril.h"
#include "base/tril.h"
#include "native/ascend/common.h"

namespace infini::ops {

// Eager triangular masking. Each call owns fresh ACL descriptors and workspace;
// exact input/output aliasing uses the dedicated inplace API.
template <>
class Operator<Tril, Device::Type::kAscend> : public Tril {
 public:
  Operator(const Tensor input, const int64_t diagonal, Tensor out)
      : Tril(input, diagonal, out) {}

  void operator()(const Tensor input, const int64_t diagonal,
                  Tensor out) const override {
    auto stream = static_cast<aclrtStream>(stream_);
    auto t_input = ascend::BuildAclTensor(input);
    auto t_out = ascend::BuildAclTensor(out);
    if (!t_input || !t_out) FailStop("descriptor creation", -1);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    const bool inplace = input.data() == out.data();
    auto ret = inplace
        ? aclnnInplaceTrilGetWorkspaceSize(t_out, diagonal, &workspace_size, &executor)
        : aclnnTrilGetWorkspaceSize(t_input, diagonal, t_out, &workspace_size, &executor);
    if (ret != ACL_SUCCESS || !executor) FailStop("workspace query", ret);
    void* workspace = nullptr;
    if (workspace_size) {
      ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_NORMAL_ONLY);
      if (ret != ACL_SUCCESS || !workspace) FailStop("workspace allocation", ret);
    }
    ret = inplace
        ? aclnnInplaceTril(workspace, workspace_size, executor, stream)
        : aclnnTril(workspace, workspace_size, executor, stream);
    if (ret != ACL_SUCCESS) FailStop("execution", ret);
    // Phase two consumes a non-repeatable executor. Retain caller descriptors
    // and backing until successful completion before releasing local resources.
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) FailStop("stream completion", ret);
    if (workspace) {
      ret = aclrtFree(workspace);
      if (ret != ACL_SUCCESS) FailStop("workspace release", ret);
    }
    ret = aclDestroyTensor(t_input);
    if (ret != ACL_SUCCESS) FailStop("input descriptor release", ret);
    ret = aclDestroyTensor(t_out);
    if (ret != ACL_SUCCESS) FailStop("output descriptor release", ret);
  }

 private:
  // ACL failures may leave executor ownership or queued work uncertain.
  // Retain resources and stop this process without freeing in-flight backing.
  [[noreturn]] static void FailStop(const char* phase, int64_t status) {
    std::fprintf(stderr, "Ascend Tril %s failed (%lld)\n",
                 phase, static_cast<long long>(status));
    std::abort();
  }
};

}  // namespace infini::ops

#endif
