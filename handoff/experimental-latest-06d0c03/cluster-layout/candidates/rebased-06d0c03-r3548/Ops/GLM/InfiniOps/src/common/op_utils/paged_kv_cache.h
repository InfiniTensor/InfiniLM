#ifndef INFINI_OPS_COMMON_OP_UTILS_PAGED_KV_CACHE_H_
#define INFINI_OPS_COMMON_OP_UTILS_PAGED_KV_CACHE_H_

#include <ATen/ATen.h>

#include <cassert>
#include <cstdint>
#include <vector>

namespace infini::ops::paged_kv_cache {

// These helpers intentionally operate on ATen tensors because linked Torch
// FlashAttention providers expose paged KV caches through ATen. Keep their use
// within linked Torch providers until a backend-independent abstraction is
// required.

inline std::vector<int32_t> ToHostInt32Vector(const at::Tensor& tensor) {
  const auto cpu = tensor.to(at::kCPU).contiguous();
  const auto* data = cpu.data_ptr<int32_t>();
  return {data, data + cpu.numel()};
}

inline at::Tensor GatherSequence(const at::Tensor& cache,
                                 const std::vector<int32_t>& block_table,
                                 int64_t table_width, int64_t batch,
                                 int64_t length) {
  if (length == 0) {
    return cache.new_empty({0, cache.size(2), cache.size(3)});
  }

  const int64_t page_size = cache.size(1);
  const int64_t block_count = (length + page_size - 1) / page_size;
  std::vector<at::Tensor> pages;
  pages.reserve(block_count);
  for (int64_t i = 0; i < block_count; ++i) {
    assert(i < table_width && "KV cache block table is too small");
    pages.push_back(cache.select(0, block_table[batch * table_width + i]));
  }
  return at::cat(pages, 0).slice(0, 0, length);
}

}  // namespace infini::ops::paged_kv_cache

#endif  // INFINI_OPS_COMMON_OP_UTILS_PAGED_KV_CACHE_H_
