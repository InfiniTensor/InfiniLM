#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_CANN82_COMPAT_HPP_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_CANN82_COMPAT_HPP_

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "aclnn_copy.h"
#include "aclnnop/level2/aclnn_rope_with_sin_cos_cache.h"
#include "base/rotary_embedding.h"
#include "native/ascend/common.h"

namespace infini::ops {

// CANN 8.2's packed-cache RoPE API accepts 2D token/head views. Its two
// outputs are staged separately so single-query calls and fused QKV views
// never alias an input while the kernel reads it. All math stays on device.
template <>
class Operator<RotaryEmbedding, Device::Type::kAscend> : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, Tensor query, std::optional<Tensor> key,
           const Tensor cos_sin_cache, int64_t head_size, bool is_neox,
           int64_t rope_dim_offset = 0, bool inverse = false)
      : RotaryEmbedding(positions, query, key, cos_sin_cache, head_size,
                        is_neox, rope_dim_offset, inverse) {
    if (rope_dim_offset != 0 || inverse)
      throw std::invalid_argument("CANN 8.2 packed-cache RoPE has no offset/inverse mode");
  }

  void operator()(const Tensor positions, Tensor query,
                  std::optional<Tensor> key, const Tensor cos_sin_cache,
                  int64_t head_size, bool is_neox, int64_t rope_dim_offset = 0,
                  bool inverse = false) const override {
    if (rope_dim_offset != 0 || inverse)
      throw std::invalid_argument("CANN 8.2 packed-cache RoPE has no offset/inverse mode");
    auto stream = static_cast<aclrtStream>(stream_);
    const auto tokens = num_tokens_;
    const auto q_width = num_heads_ * head_size_;
    const auto k_width = key ? num_kv_heads_ * head_size_ : q_width;
    const Tensor q_input{query.data(), {tokens, q_width}, query.dtype(), query.device(), {query_token_stride_, 1}};
    const Tensor k_input{key ? key->data() : query.data(), {tokens, k_width}, query.dtype(), query.device(), {key ? key_token_stride_ : query_token_stride_, 1}};
    const Tensor pos{const_cast<void*>(positions.data()), {tokens}, DataType::kInt64, positions.device(), {1}};
    void* q_data = nullptr;
    void* k_data = nullptr;
    const auto q_bytes = tokens * q_width * query.element_size();
    const auto k_bytes = tokens * k_width * query.element_size();
    Check(aclrtMalloc(&q_data, q_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "query staging allocation");
    Check(aclrtMalloc(&k_data, k_bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "key staging allocation");
    if (!q_data || !k_data) FailStop("null staging allocation", -1);
    const Tensor q_output{q_data, {tokens, q_width}, query.dtype(), query.device()};
    const Tensor k_output{k_data, {tokens, k_width}, query.dtype(), query.device()};
    auto t_pos = ascend::BuildAclTensor(pos);
    auto t_q = ascend::BuildAclTensor(q_input);
    auto t_k = ascend::BuildAclTensor(k_input);
    auto t_cache = ascend::BuildAclTensor(cos_sin_cache);
    auto t_q_out = ascend::BuildAclTensor(q_output);
    auto t_k_out = ascend::BuildAclTensor(k_output);
    for (auto tensor : {t_pos, t_q, t_k, t_cache, t_q_out, t_k_out})
      if (!tensor) FailStop("descriptor creation", -1);
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    Check(aclnnRopeWithSinCosCacheGetWorkspaceSize(t_pos, t_q, t_k, t_cache,
          nullptr, head_size, is_neox, t_q_out, t_k_out, &workspace_size, &executor), "RoPE workspace query");
    if (!executor) FailStop("null RoPE executor", -1);
    void* workspace = AllocateWorkspace(workspace_size);
    Check(aclnnRopeWithSinCosCache(workspace, workspace_size, executor, stream), "RoPE execution");
    Check(aclrtSynchronizeStream(stream), "RoPE completion");
    if (workspace) Check(aclrtFree(workspace), "RoPE workspace release");
    CopyBack(t_q, t_q_out, stream);
    if (key) CopyBack(t_k, t_k_out, stream);
    for (auto tensor : {t_pos, t_q, t_k, t_cache, t_q_out, t_k_out})
      Check(aclDestroyTensor(tensor), "descriptor release");
    Check(aclrtFree(q_data), "query staging release");
    Check(aclrtFree(k_data), "key staging release");
  }

 private:
  static void* AllocateWorkspace(uint64_t size) {
    void* data = nullptr;
    if (size) {
      Check(aclrtMalloc(&data, size, ACL_MEM_MALLOC_NORMAL_ONLY), "workspace allocation");
      if (!data) FailStop("null workspace allocation", -1);
    }
    return data;
  }

  static void CopyBack(aclTensor* destination, aclTensor* source, aclrtStream stream) {
    uint64_t size = 0;
    aclOpExecutor* executor = nullptr;
    Check(aclnnInplaceCopyGetWorkspaceSize(destination, source, &size, &executor), "copy workspace query");
    if (!executor) FailStop("null copy executor", -1);
    void* workspace = AllocateWorkspace(size);
    Check(aclnnInplaceCopy(workspace, size, executor, stream), "copy execution");
    Check(aclrtSynchronizeStream(stream), "copy completion");
    if (workspace) Check(aclrtFree(workspace), "copy workspace release");
  }

  static void Check(int64_t status, const char* phase) {
    if (status != ACL_SUCCESS) FailStop(phase, status);
  }

  [[noreturn]] static void FailStop(const char* phase, int64_t status) {
    std::fprintf(stderr, "Ascend RotaryEmbedding %s failed (%lld)\n", phase,
                 static_cast<long long>(status));
    std::abort();
  }
};

}  // namespace infini::ops

#endif
