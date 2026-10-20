#ifndef INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
#define INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_FLASH_KERNEL_H_

#include <array>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnnop/aclnn_scatter_nd_update.h"
#include "aclnnop/aclnn_scatter_pa_kv_cache.h"
#include "base/reshape_and_cache_flash.h"
#include "data_type.h"
#include "native/ascend/common.h"
#include "native/ascend/workspace_pool_.h"
#include "operator.h"

#ifdef INFINI_HAS_CUSTOM_KERNELS
#include "aclrtlaunch_reshape_and_cache_flash_strided.h"
#endif

namespace infini::ops {

template <>
class Operator<ReshapeAndCacheFlash, Device::Type::kAscend>
    : public ReshapeAndCacheFlash {
 public:
  Operator(const Tensor key, const Tensor value, const Tensor slot_mapping,
           const Tensor k_scale, const Tensor v_scale,
           const std::string kv_cache_dtype, Tensor key_cache,
           Tensor value_cache)
      : ReshapeAndCacheFlash(key, value, slot_mapping, k_scale, v_scale,
                             kv_cache_dtype, key_cache, value_cache),
        use_fast_path_(dtype_ != DataType::kFloat32 && key.IsContiguous() &&
                       value.IsContiguous() && key_cache.IsContiguous() &&
                       value_cache.IsContiguous()),
        num_blocks_(key_cache.size(0)),
        element_size_(kDataTypeToSize.at(dtype_)),
        fast_key_cache_(PrefixView(key, num_tokens_)),
        fast_value_cache_(PrefixView(value, num_tokens_)),
        key_update_cache_(PrefixView(key, 1)),
        value_update_cache_(PrefixView(value, 1)),
        slot_mapping_cache_(slot_mapping),
        indices_cache_({1, 2}, ACL_INT64, nullptr),
        key_cache_cache_(key_cache),
        value_cache_cache_(value_cache),
        slot_mapping_host_(num_tokens_),
        indices_host_(num_tokens_ * 2) {
    if (!use_fast_path_ && num_tokens_ > 0) {
      auto ret = aclrtMalloc(&indices_device_,
                             indices_host_.size() * sizeof(indices_host_[0]),
                             ACL_MEM_MALLOC_NORMAL_ONLY);
      assert(ret == ACL_SUCCESS &&
             "Ascend `ReshapeAndCacheFlash` failed to allocate indices");
    }
  }

  ~Operator() override {
    if (!ascend::IsAclRuntimeAlive()) return;

    if (indices_device_) aclrtFree(indices_device_);
  }

  void operator()(const Tensor key, const Tensor value,
                  const Tensor slot_mapping, const Tensor /*k_scale*/,
                  const Tensor /*v_scale*/,
                  const std::string /*kv_cache_dtype*/, Tensor key_cache,
                  Tensor value_cache) const override {
    if (num_tokens_ == 0) return;

    auto stream = static_cast<aclrtStream>(stream_);
    if (use_fast_path_) {
      RunFastPath(key, value, slot_mapping, key_cache, value_cache, stream);
    } else {
#ifdef INFINI_HAS_CUSTOM_KERNELS
      RunStridedKernel(key, value, slot_mapping, key_cache, value_cache,
                       stream);
#else
      RunFallback(key, value, slot_mapping, key_cache, value_cache, stream);
#endif
    }
  }

 private:
  static Tensor PrefixView(const Tensor tensor, Tensor::Size size) {
    auto shape = Tensor::Shape{tensor.shape()};
    shape[0] = size;
    return Tensor{const_cast<void*>(tensor.data()), shape, tensor.dtype(),
                  tensor.device(), tensor.strides()};
  }

  void RunFastPath(const Tensor key, const Tensor value,
                   const Tensor slot_mapping, Tensor key_cache,
                   Tensor value_cache, aclrtStream stream) const {
    auto t_key = fast_key_cache_.get(const_cast<void*>(key.data()));
    auto t_value = fast_value_cache_.get(const_cast<void*>(value.data()));
    auto t_slot_mapping =
        slot_mapping_cache_.get(const_cast<void*>(slot_mapping.data()));
    auto t_key_cache = key_cache_cache_.get(key_cache.data());
    auto t_value_cache = value_cache_cache_.get(value_cache.data());

    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;
    auto ret = aclnnScatterPaKvCacheGetWorkspaceSize(
        t_key, t_key_cache, t_slot_mapping, t_value, t_value_cache, nullptr,
        nullptr, nullptr, cache_mode_.data(), nullptr, nullptr, nullptr,
        &workspace_size, &executor);
    assert(ret == ACL_SUCCESS &&
           "Ascend `ReshapeAndCacheFlash` workspace query failed");

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    ret = aclnnScatterPaKvCache(arena.buf, workspace_size, executor, stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `ReshapeAndCacheFlash` execution failed");
  }

#ifdef INFINI_HAS_CUSTOM_KERNELS
  void RunStridedKernel(const Tensor key, const Tensor value,
                        const Tensor slot_mapping, Tensor key_cache,
                        Tensor value_cache, aclrtStream stream) const {
    auto ret = aclrtlaunch_reshape_and_cache_flash_strided(
        1, stream, const_cast<void*>(key.data()),
        const_cast<void*>(value.data()), const_cast<void*>(slot_mapping.data()),
        key_cache.data(), value_cache.data(), static_cast<int64_t>(num_tokens_),
        static_cast<int64_t>(num_heads_), static_cast<int64_t>(head_size_),
        static_cast<int64_t>(block_size_), static_cast<int64_t>(num_blocks_),
        static_cast<int64_t>(key_token_stride_),
        static_cast<int64_t>(key_head_stride_),
        static_cast<int64_t>(value_token_stride_),
        static_cast<int64_t>(value_head_stride_),
        static_cast<int64_t>(key_cache_block_stride_),
        static_cast<int64_t>(key_cache_page_stride_),
        static_cast<int64_t>(key_cache_head_stride_),
        static_cast<int64_t>(value_cache_block_stride_),
        static_cast<int64_t>(value_cache_page_stride_),
        static_cast<int64_t>(value_cache_head_stride_),
        static_cast<int64_t>(element_size_));
    assert(ret == ACL_SUCCESS &&
           "Ascend `ReshapeAndCacheFlash` strided execution failed");
  }
#endif

  void RunFallback(const Tensor key, const Tensor value,
                   const Tensor slot_mapping, Tensor key_cache,
                   Tensor value_cache, aclrtStream stream) const {
    auto mapping_bytes = num_tokens_ * sizeof(slot_mapping_host_[0]);
    auto ret = aclrtMemcpyAsync(slot_mapping_host_.data(), mapping_bytes,
                                slot_mapping.data(), mapping_bytes,
                                ACL_MEMCPY_DEVICE_TO_HOST, stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `ReshapeAndCacheFlash` failed to copy slot mapping");
    ret = aclrtSynchronizeStream(stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `ReshapeAndCacheFlash` failed to synchronize slot mapping");

    for (std::size_t token = 0; token < num_tokens_; ++token) {
      auto slot = slot_mapping_host_[token];
      if (slot < 0) continue;

      assert(static_cast<std::size_t>(slot) < num_blocks_ * block_size_ &&
             "Ascend `ReshapeAndCacheFlash` slot is outside the cache");
      indices_host_[token * 2] = slot / block_size_;
      indices_host_[token * 2 + 1] = slot % block_size_;
    }

    auto indices_bytes = indices_host_.size() * sizeof(indices_host_[0]);
    ret = aclrtMemcpyAsync(indices_device_, indices_bytes, indices_host_.data(),
                           indices_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `ReshapeAndCacheFlash` failed to copy cache indices");

    auto* key_data = static_cast<const char*>(key.data());
    auto* value_data = static_cast<const char*>(value.data());
    auto* indices_data = static_cast<char*>(indices_device_);
    auto t_key_cache = key_cache_cache_.get(key_cache.data());
    auto t_value_cache = value_cache_cache_.get(value_cache.data());

    for (std::size_t token = 0; token < num_tokens_; ++token) {
      if (slot_mapping_host_[token] < 0) continue;

      auto key_offset = token * key_token_stride_ * element_size_;
      auto value_offset = token * value_token_stride_ * element_size_;
      auto indices_offset = token * 2 * sizeof(indices_host_[0]);
      auto t_key_update =
          key_update_cache_.get(const_cast<char*>(key_data + key_offset));
      auto t_value_update =
          value_update_cache_.get(const_cast<char*>(value_data + value_offset));
      auto t_indices = indices_cache_.get(indices_data + indices_offset);

      ScatterUpdate(t_key_cache, t_indices, t_key_update, stream);
      ScatterUpdate(t_value_cache, t_indices, t_value_update, stream);
    }
  }

  static void ScatterUpdate(aclTensor* cache, const aclTensor* indices,
                            const aclTensor* update, aclrtStream stream) {
    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;
    auto ret = aclnnScatterNdUpdateGetWorkspaceSize(cache, indices, update,
                                                    &workspace_size, &executor);
    assert(ret == ACL_SUCCESS &&
           "Ascend `ReshapeAndCacheFlash` fallback query failed");

    auto& arena = ascend::GetWorkspacePool().Ensure(stream, workspace_size);
    ret = aclnnScatterNdUpdate(arena.buf, workspace_size, executor, stream);
    assert(ret == ACL_SUCCESS &&
           "Ascend `ReshapeAndCacheFlash` fallback execution failed");
  }

  bool use_fast_path_{false};
  std::size_t num_blocks_{0};
  std::size_t element_size_{0};

  mutable ascend::AclTensorCache fast_key_cache_;
  mutable ascend::AclTensorCache fast_value_cache_;
  mutable ascend::AclTensorCache key_update_cache_;
  mutable ascend::AclTensorCache value_update_cache_;
  mutable ascend::AclTensorCache slot_mapping_cache_;
  mutable ascend::AclTensorCache indices_cache_;
  mutable ascend::AclTensorCache key_cache_cache_;
  mutable ascend::AclTensorCache value_cache_cache_;

  mutable std::array<char, 5> cache_mode_{'N', 'o', 'r', 'm', '\0'};
  void* indices_device_{nullptr};
  mutable std::vector<int64_t> slot_mapping_host_;
  mutable std::vector<int64_t> indices_host_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_ASCEND_RESHAPE_AND_CACHE_FLASH_KERNEL_H_
