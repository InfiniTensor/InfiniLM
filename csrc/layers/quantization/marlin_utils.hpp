#pragma once

#include "infinicore/tensor.hpp"

#include <cstdint>
#include <vector>

namespace infinilm::quantization::marlin {

constexpr int64_t UINT4_ID = 1125899906843648LL;
constexpr int64_t UINT4B8_ID = 1125899907892224LL;

bool supports_shape(size_t input_size_per_partition, size_t output_size_per_partition, int group_size);

infinicore::Tensor make_empty_i32(const infinicore::Device &device);
infinicore::Tensor make_i32_tensor(const std::vector<int32_t> &data, const std::vector<size_t> &shape, const infinicore::Device &device);

infinicore::Tensor awq_marlin_repack(const infinicore::Tensor &qweight, size_t size_k, size_t size_n, int num_bits);
infinicore::Tensor gptq_marlin_repack(const infinicore::Tensor &qweight, const infinicore::Tensor &perm, size_t size_k, size_t size_n, int num_bits);

infinicore::Tensor sort_g_idx(const infinicore::Tensor &g_idx, infinicore::Tensor &sort_indices);
infinicore::Tensor permute_scales(const infinicore::Tensor &scales, size_t size_k, size_t size_n, int group_size);
infinicore::Tensor awq_to_marlin_zero_points(const infinicore::Tensor &qzeros, size_t size_k, size_t size_n, int num_bits);
infinicore::Tensor permute_bias(const infinicore::Tensor &bias);

} // namespace infinilm::quantization::marlin
