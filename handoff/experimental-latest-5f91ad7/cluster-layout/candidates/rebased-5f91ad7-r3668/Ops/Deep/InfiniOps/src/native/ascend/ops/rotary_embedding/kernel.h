#ifndef INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_
#define INFINI_OPS_ASCEND_ROTARY_EMBEDDING_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_copy.h"
#include "aclnnop/aclnn_index_select.h"
#include "aclnnop/aclnn_rotary_position_embedding.h"
#include "base/rotary_embedding.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

namespace infini::ops {

// NeoX/GPT-J RoPE. Positions select rows from the packed [cos, sin] cache,
// then CANN rotates the configured prefix of query and an optional key.
template <>
class Operator<RotaryEmbedding, Device::Type::kAscend>
    : public RotaryEmbedding {
 public:
  Operator(const Tensor positions, Tensor query, std::optional<Tensor> key,
           const Tensor cos_sin_cache, int64_t head_size, bool is_neox,
           int64_t rope_dim_offset = 0, bool inverse = false)
      : RotaryEmbedding(positions, query, key, cos_sin_cache, head_size,
                        is_neox, rope_dim_offset, inverse),
        max_seq_len_(static_cast<int64_t>(cos_sin_cache.size(0))),
        element_size_(cos_sin_cache.element_size()),
        has_key_(key.has_value()),
        rotary_mode_(is_neox ? 0 : 1) {
    assert(rope_dim_offset_ == 0 && !inverse_ && rot_dim_ <= head_size_ &&
           (!is_neox_ || rot_dim_ == head_size_) &&
           "Ascend `RotaryEmbedding` supports full-dimension forward "
           "NeoX/GPT-J rotation and partial-prefix forward GPT-J rotation");
    assert(cos_sin_cache_type_ == query_type_ &&
           "Ascend `RotaryEmbedding` requires cache and query dtypes to "
           "match");
    assert(cos_sin_cache.IsContiguous() &&
           "Ascend `RotaryEmbedding` requires a contiguous cache tensor");

    const auto num_tokens = static_cast<int64_t>(num_tokens_);
    const auto head_dim = head_size_;
    const auto acl_dtype = ascend::ToAclDtype(query.dtype());
    const auto table_bytes =
        static_cast<size_t>(max_seq_len_ * head_dim) * element_size_;
    auto ret =
        aclrtMalloc(&cos_table_data_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS &&
           "Ascend `RotaryEmbedding` failed to allocate cosine table");
    ret =
        aclrtMalloc(&sin_table_data_, table_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS &&
           "Ascend `RotaryEmbedding` failed to allocate sine table");

    UploadCosSinCache(cos_sin_cache);
    cos_sin_cache_data_ = cos_sin_cache.data();

    const auto gathered_bytes =
        static_cast<size_t>(num_tokens * head_dim) * element_size_;
    ret = aclrtMalloc(&cos_data_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS &&
           "Ascend `RotaryEmbedding` failed to allocate gathered cosine");
    ret = aclrtMalloc(&sin_data_, gathered_bytes, ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS &&
           "Ascend `RotaryEmbedding` failed to allocate gathered sine");

    cos_table_cache_ = ascend::AclTensorCache({max_seq_len_, head_dim},
                                              acl_dtype, cos_table_data_);
    sin_table_cache_ = ascend::AclTensorCache({max_seq_len_, head_dim},
                                              acl_dtype, sin_table_data_);
    positions_cache_ = ascend::AclTensorCache(
        {num_tokens}, ACL_INT64, const_cast<void*>(positions.data()));
    cos_out_cache_ =
        ascend::AclTensorCache({num_tokens, head_dim}, acl_dtype, cos_data_);
    sin_out_cache_ =
        ascend::AclTensorCache({num_tokens, head_dim}, acl_dtype, sin_data_);
    cos_rotary_cache_ =
        ascend::AclTensorCache({num_tokens, 1, head_dim}, acl_dtype, cos_data_);
    sin_rotary_cache_ =
        ascend::AclTensorCache({num_tokens, 1, head_dim}, acl_dtype, sin_data_);
    query_cache_ = ascend::AclTensorCache(
        Tensor{query.data(),
               {num_tokens, static_cast<int64_t>(num_heads_), head_dim},
               query.dtype(),
               query.device(),
               {query_token_stride_, query_head_stride_, 1}});

    query_bytes_ = static_cast<size_t>(query.numel()) * element_size_;
    ret =
        aclrtMalloc(&query_out_data_, query_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);
    assert(ret == ACL_SUCCESS &&
           "Ascend `RotaryEmbedding` failed to allocate query output");
    query_out_cache_ = ascend::AclTensorCache(
        {num_tokens, static_cast<int64_t>(num_heads_), head_dim}, acl_dtype,
        query_out_data_);

    if (key.has_value()) {
      key_cache_ = ascend::AclTensorCache(
          Tensor{key->data(),
                 {num_tokens, static_cast<int64_t>(num_kv_heads_), head_dim},
                 key->dtype(),
                 key->device(),
                 {key_token_stride_, key_head_stride_, 1}});
      key_bytes_ = static_cast<size_t>(key->numel()) * element_size_;
      ret = aclrtMalloc(&key_out_data_, key_bytes_, ACL_MEM_MALLOC_NORMAL_ONLY);
      assert(ret == ACL_SUCCESS &&
             "Ascend `RotaryEmbedding` failed to allocate key output");
      key_out_cache_ = ascend::AclTensorCache(
          {num_tokens, static_cast<int64_t>(num_kv_heads_), head_dim},
          acl_dtype, key_out_data_);
    }
  }

  ~Operator() {
    if (!ascend::IsAclRuntimeAlive()) return;

    cos_table_cache_.release();
    sin_table_cache_.release();
    positions_cache_.release();
    cos_out_cache_.release();
    sin_out_cache_.release();
    cos_rotary_cache_.release();
    sin_rotary_cache_.release();
    query_cache_.release();
    key_cache_.release();
    query_out_cache_.release();
    key_out_cache_.release();
    if (cos_table_data_) aclrtFree(cos_table_data_);
    if (sin_table_data_) aclrtFree(sin_table_data_);
    if (cos_data_) aclrtFree(cos_data_);
    if (sin_data_) aclrtFree(sin_data_);
    if (query_out_data_) aclrtFree(query_out_data_);
    if (key_out_data_) aclrtFree(key_out_data_);
  }

  void operator()(const Tensor positions, Tensor query,
                  std::optional<Tensor> key, const Tensor cos_sin_cache,
                  int64_t head_size, bool is_neox, int64_t rope_dim_offset = 0,
                  bool inverse = false) const override {
    assert(key.has_value() == has_key_ &&
           "Ascend `RotaryEmbedding` key presence changed after planning");
    auto stream = static_cast<aclrtStream>(stream_);

    if (cos_sin_cache.data() != cos_sin_cache_data_) {
      UploadCosSinCache(cos_sin_cache);
      cos_sin_cache_data_ = cos_sin_cache.data();
    }

    auto t_cos_table = cos_table_cache_.get(cos_table_data_);
    auto t_sin_table = sin_table_cache_.get(sin_table_data_);
    auto t_positions =
        positions_cache_.get(const_cast<void*>(positions.data()));
    auto t_cos_out = cos_out_cache_.get(cos_data_);
    auto t_sin_out = sin_out_cache_.get(sin_data_);

    if (!cos_index_executor_) {
      aclnnIndexSelectGetWorkspaceSize(t_cos_table, 0, t_positions, t_cos_out,
                                       &cos_index_ws_size_,
                                       &cos_index_executor_);
      aclSetAclOpExecutorRepeatable(cos_index_executor_);
    } else {
      aclSetInputTensorAddr(cos_index_executor_, 1, t_positions,
                            const_cast<void*>(positions.data()));
    }

    if (!sin_index_executor_) {
      aclnnIndexSelectGetWorkspaceSize(t_sin_table, 0, t_positions, t_sin_out,
                                       &sin_index_ws_size_,
                                       &sin_index_executor_);
      aclSetAclOpExecutorRepeatable(sin_index_executor_);
    } else {
      aclSetInputTensorAddr(sin_index_executor_, 1, t_positions,
                            const_cast<void*>(positions.data()));
    }

    auto index_ws_size = std::max(cos_index_ws_size_, sin_index_ws_size_);
    auto& index_arena =
        ascend::GetWorkspacePool().Ensure(stream, index_ws_size);
    aclnnIndexSelect(index_arena.buf, cos_index_ws_size_, cos_index_executor_,
                     stream);
    aclnnIndexSelect(index_arena.buf, sin_index_ws_size_, sin_index_executor_,
                     stream);

    auto t_cos = cos_rotary_cache_.get(cos_data_);
    auto t_sin = sin_rotary_cache_.get(sin_data_);
    auto t_query = query_cache_.get(query.data());
    RunOne(t_query, t_cos, t_sin, query.data(), query_out_cache_,
           query_out_data_, query_rotary_executor_, query_rotary_ws_size_,
           stream);

    if (key.has_value()) {
      auto t_key = key_cache_.get(key->data());
      RunOne(t_key, t_cos, t_sin, key->data(), key_out_cache_, key_out_data_,
             key_rotary_executor_, key_rotary_ws_size_, stream);
    }
  }

 private:
  void RunOne(aclTensor* input, aclTensor* cos, aclTensor* sin,
              void* input_data, ascend::AclTensorCache& output_cache,
              void* output_data, aclOpExecutor*& executor,
              uint64_t& workspace_size, aclrtStream stream) const {
    auto output = output_cache.get(output_data);
    if (!executor) {
      auto ret = aclnnRotaryPositionEmbeddingGetWorkspaceSize(
          input, cos, sin, rotary_mode_, output, &workspace_size, &executor);
      assert(ret == ACL_SUCCESS &&
             "`aclnnRotaryPositionEmbeddingGetWorkspaceSize` failed");
      aclSetAclOpExecutorRepeatable(executor);
    } else {
      aclSetInputTensorAddr(executor, 0, input, input_data);
      aclSetInputTensorAddr(executor, 1, cos, cos_data_);
      aclSetInputTensorAddr(executor, 2, sin, sin_data_);
      aclSetOutputTensorAddr(executor, 0, output, output_data);
    }

    auto& rotary_arena =
        ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    auto ret = aclnnRotaryPositionEmbedding(rotary_arena.buf, workspace_size,
                                            executor, stream);
    assert(ret == ACL_SUCCESS && "`aclnnRotaryPositionEmbedding` failed");

    // Q/K can be narrow views into a fused QKV projection. Preserve their
    // real token strides when copying the contiguous ACLNN output back.
    aclOpExecutor* copy_executor = nullptr;
    uint64_t copy_workspace_size = 0;
    ret = aclnnInplaceCopyGetWorkspaceSize(input, output, &copy_workspace_size,
                                           &copy_executor);
    assert(ret == ACL_SUCCESS &&
           "Planning Ascend `RotaryEmbedding` output copy failed");
    auto& copy_arena =
        ascend::GetWorkspacePool().Ensure(stream, copy_workspace_size);
    ret = aclnnInplaceCopy(copy_arena.buf, copy_workspace_size, copy_executor,
                           stream);
    assert(ret == ACL_SUCCESS &&
           "Copying Ascend `RotaryEmbedding` output failed");
  }

  void UploadCosSinCache(const Tensor cos_sin_cache) const {
    const auto half_dim = rot_dim_ / 2;
    const auto packed_bytes =
        static_cast<size_t>(max_seq_len_ * rot_dim_) * element_size_;
    const auto table_bytes =
        static_cast<size_t>(max_seq_len_ * head_size_) * element_size_;
    std::vector<uint8_t> packed(packed_bytes);
    std::vector<uint8_t> cosine(table_bytes);
    std::vector<uint8_t> sine(table_bytes);

    auto ret = aclrtMemcpy(packed.data(), packed_bytes, cos_sin_cache.data(),
                           packed_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    assert(ret == ACL_SUCCESS &&
           "Ascend `RotaryEmbedding` failed to read cos/sin cache");
    for (int64_t position = 0; position < max_seq_len_; ++position) {
      for (int64_t index = 0; index < head_size_; ++index) {
        const auto destination_index = position * head_size_ + index;
        WriteOne(cosine.data() + destination_index * element_size_);
      }
      for (int64_t index = 0; index < rot_dim_; ++index) {
        const auto source_index = is_neox_ ? index % half_dim : index / 2;
        const auto cos_source =
            packed.data() +
            (position * rot_dim_ + source_index) * element_size_;
        const auto sin_source =
            packed.data() +
            (position * rot_dim_ + half_dim + source_index) * element_size_;
        const auto destination_index = position * head_size_ + index;
        std::memcpy(cosine.data() + destination_index * element_size_,
                    cos_source, element_size_);
        std::memcpy(sine.data() + destination_index * element_size_, sin_source,
                    element_size_);
      }
    }

    ret = aclrtMemcpy(cos_table_data_, table_bytes, cosine.data(), table_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "Ascend `RotaryEmbedding` failed to upload cosine table");
    ret = aclrtMemcpy(sin_table_data_, table_bytes, sine.data(), table_bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    assert(ret == ACL_SUCCESS &&
           "Ascend `RotaryEmbedding` failed to upload sine table");
  }

  void WriteOne(uint8_t* destination) const {
    if (query_type_ == DataType::kFloat16) {
      const uint16_t one = 0x3c00;
      std::memcpy(destination, &one, sizeof(one));
    } else if (query_type_ == DataType::kBFloat16) {
      const uint16_t one = 0x3f80;
      std::memcpy(destination, &one, sizeof(one));
    } else {
      const float one = 1.0f;
      std::memcpy(destination, &one, sizeof(one));
    }
  }

  int64_t max_seq_len_{0};
  size_t element_size_{0};
  bool has_key_{false};
  int64_t rotary_mode_{0};
  size_t query_bytes_{0};
  size_t key_bytes_{0};
  mutable const void* cos_sin_cache_data_{nullptr};
  void* cos_table_data_{nullptr};
  void* sin_table_data_{nullptr};
  void* cos_data_{nullptr};
  void* sin_data_{nullptr};
  void* query_out_data_{nullptr};
  void* key_out_data_{nullptr};
  mutable ascend::AclTensorCache cos_table_cache_;
  mutable ascend::AclTensorCache sin_table_cache_;
  mutable ascend::AclTensorCache positions_cache_;
  mutable ascend::AclTensorCache cos_out_cache_;
  mutable ascend::AclTensorCache sin_out_cache_;
  mutable ascend::AclTensorCache cos_rotary_cache_;
  mutable ascend::AclTensorCache sin_rotary_cache_;
  mutable ascend::AclTensorCache query_cache_;
  mutable ascend::AclTensorCache key_cache_;
  mutable ascend::AclTensorCache query_out_cache_;
  mutable ascend::AclTensorCache key_out_cache_;
  mutable aclOpExecutor* cos_index_executor_{nullptr};
  mutable uint64_t cos_index_ws_size_{0};
  mutable aclOpExecutor* sin_index_executor_{nullptr};
  mutable uint64_t sin_index_ws_size_{0};
  mutable aclOpExecutor* query_rotary_executor_{nullptr};
  mutable uint64_t query_rotary_ws_size_{0};
  mutable aclOpExecutor* key_rotary_executor_{nullptr};
  mutable uint64_t key_rotary_ws_size_{0};
};

}  // namespace infini::ops

#endif
