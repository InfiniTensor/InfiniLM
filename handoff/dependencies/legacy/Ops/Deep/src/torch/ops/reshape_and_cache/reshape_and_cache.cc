#include "torch/ops/reshape_and_cache/reshape_and_cache.h"

#include "torch/tensor_.h"

namespace infini::ops {
namespace {

at::Tensor PrepareCacheValues(const at::Tensor& values, const at::Tensor& scale,
                              const std::string& kv_cache_dtype) {
  if (kv_cache_dtype == "auto") {
    return values;
  }

  const auto fp8_type =
      kv_cache_dtype == "fp8_e5m2" ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
  return (values / scale).to(fp8_type).view(at::kByte);
}

}  // namespace

template <Device::Type kDev>
Operator<ReshapeAndCache, kDev, 1>::Operator(
    const Tensor key, const Tensor value, Tensor key_cache, Tensor value_cache,
    const Tensor slot_mapping, const Tensor k_scale, const Tensor v_scale,
    const std::string kv_cache_dtype)
    : ReshapeAndCache{key,          value,   key_cache, value_cache,
                      slot_mapping, k_scale, v_scale,   kv_cache_dtype} {}

template <Device::Type kDev>
void Operator<ReshapeAndCache, kDev, 1>::operator()(
    const Tensor key, const Tensor value, Tensor key_cache, Tensor value_cache,
    const Tensor slot_mapping, const Tensor k_scale, const Tensor v_scale,
    const std::string kv_cache_dtype) const {
  auto at_key = ToAtenTensor<kDev>(const_cast<void*>(key.data()), key_shape_,
                                   key_strides_, key_type_, device_index_);
  auto at_value =
      ToAtenTensor<kDev>(const_cast<void*>(value.data()), value_shape_,
                         value_strides_, key_type_, device_index_);
  auto at_key_cache =
      ToAtenTensor<kDev>(key_cache.data(), key_cache_shape_, key_cache_strides_,
                         key_cache_type_, device_index_);
  auto at_value_cache = ToAtenTensor<kDev>(
      value_cache.data(), value_cache_shape_, value_cache_strides_,
      value_cache_type_, device_index_);
  auto at_slot_mapping = ToAtenTensor<kDev>(
      const_cast<void*>(slot_mapping.data()), slot_mapping_shape_,
      slot_mapping_strides_, DataType::kInt64, device_index_);
  auto at_k_scale =
      ToAtenTensor<kDev>(const_cast<void*>(k_scale.data()), k_scale_shape_,
                         k_scale_strides_, k_scale_type_, device_index_);
  auto at_v_scale =
      ToAtenTensor<kDev>(const_cast<void*>(v_scale.data()), v_scale_shape_,
                         v_scale_strides_, v_scale_type_, device_index_);

  auto token_indices = at::nonzero(at_slot_mapping >= 0).reshape({-1});
  if (token_indices.numel() == 0) {
    return;
  }

  auto slots = at_slot_mapping.index_select(0, token_indices);
  auto block_indices =
      at::floor_divide(slots, static_cast<int64_t>(block_size_));
  auto block_offsets = at::remainder(slots, static_cast<int64_t>(block_size_));
  auto key_values =
      at_key.index_select(0, token_indices)
          .view({token_indices.numel(), static_cast<int64_t>(num_heads_),
                 static_cast<int64_t>(head_size_ / x_),
                 static_cast<int64_t>(x_)});
  auto value_values = at_value.index_select(0, token_indices);
  key_values = PrepareCacheValues(key_values, at_k_scale, kv_cache_dtype);
  value_values = PrepareCacheValues(value_values, at_v_scale, kv_cache_dtype);

  using torch::indexing::Slice;
  at_key_cache.index_put_(
      {block_indices, Slice(), Slice(), block_offsets, Slice()}, key_values);
  at_value_cache.index_put_({block_indices, Slice(), Slice(), block_offsets},
                            value_values);
}

template class Operator<ReshapeAndCache, Device::Type::kCpu, 1>;
template class Operator<ReshapeAndCache, Device::Type::kNvidia, 1>;
template class Operator<ReshapeAndCache, Device::Type::kCambricon, 1>;
template class Operator<ReshapeAndCache, Device::Type::kAscend, 1>;
template class Operator<ReshapeAndCache, Device::Type::kMetax, 1>;
template class Operator<ReshapeAndCache, Device::Type::kMoore, 1>;
template class Operator<ReshapeAndCache, Device::Type::kIluvatar, 1>;
template class Operator<ReshapeAndCache, Device::Type::kHygon, 1>;
template class Operator<ReshapeAndCache, Device::Type::kMars, 1>;

}  // namespace infini::ops
