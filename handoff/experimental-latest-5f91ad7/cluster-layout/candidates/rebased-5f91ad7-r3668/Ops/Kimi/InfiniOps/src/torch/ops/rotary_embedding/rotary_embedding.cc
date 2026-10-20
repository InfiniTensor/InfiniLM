#include "torch/ops/rotary_embedding/rotary_embedding.h"

#include "torch/tensor_.h"

namespace infini::ops {
namespace {

void ApplyRotaryEmbedding(at::Tensor data, const at::Tensor& positions,
                          const at::Tensor& cos_sin_cache, int64_t token_stride,
                          int64_t head_stride, int64_t num_heads,
                          int64_t head_size, int64_t rot_dim,
                          int64_t rope_dim_offset, bool is_neox, bool inverse) {
  const int64_t num_tokens = positions.numel();
  const int64_t embed_dim = rot_dim / 2;
  auto data_view = data.as_strided({num_tokens, num_heads, head_size},
                                   {token_stride, head_stride, 1});
  auto cache = cos_sin_cache.index_select(0, positions.reshape({-1}));
  auto cos = cache.slice(1, 0, embed_dim).unsqueeze(1).to(at::kFloat);
  auto sin = cache.slice(1, embed_dim, rot_dim).unsqueeze(1).to(at::kFloat);
  if (inverse) {
    sin = -sin;
  }

  auto rotary = data_view.slice(2, rope_dim_offset, rope_dim_offset + rot_dim);
  if (is_neox) {
    auto x = rotary.slice(2, 0, embed_dim).to(at::kFloat).clone();
    auto y = rotary.slice(2, embed_dim, rot_dim).to(at::kFloat).clone();
    rotary.slice(2, 0, embed_dim).copy_(x * cos - y * sin);
    rotary.slice(2, embed_dim, rot_dim).copy_(y * cos + x * sin);
  } else {
    auto x = rotary.slice(2, 0, rot_dim, 2).to(at::kFloat).clone();
    auto y = rotary.slice(2, 1, rot_dim, 2).to(at::kFloat).clone();
    rotary.slice(2, 0, rot_dim, 2).copy_(x * cos - y * sin);
    rotary.slice(2, 1, rot_dim, 2).copy_(y * cos + x * sin);
  }
}

}  // namespace

template <Device::Type kDev>
Operator<RotaryEmbedding, kDev, 1>::Operator(
    const Tensor positions, Tensor query, std::optional<Tensor> key,
    const Tensor cos_sin_cache, int64_t head_size, bool is_neox,
    int64_t rope_dim_offset, bool inverse)
    : RotaryEmbedding{positions,       query,     key,
                      cos_sin_cache,   head_size, is_neox,
                      rope_dim_offset, inverse} {}

template <Device::Type kDev>
void Operator<RotaryEmbedding, kDev, 1>::operator()(
    const Tensor positions, Tensor query, std::optional<Tensor> key,
    const Tensor cos_sin_cache, int64_t, bool, int64_t, bool) const {
  if (num_tokens_ == 0) {
    return;
  }

  auto at_positions =
      ToAtenTensor<kDev>(const_cast<void*>(positions.data()), positions_shape_,
                         positions_strides_, positions_type_, device_index_);
  auto at_query = ToAtenTensor<kDev>(query.data(), query_shape_, query_strides_,
                                     query_type_, device_index_);
  auto at_cache = ToAtenTensor<kDev>(
      const_cast<void*>(cos_sin_cache.data()), cos_sin_cache_shape_,
      cos_sin_cache_strides_, cos_sin_cache_type_, device_index_);

  ApplyRotaryEmbedding(at_query, at_positions, at_cache, query_token_stride_,
                       query_head_stride_, num_heads_, head_size_, rot_dim_,
                       rope_dim_offset_, is_neox_, inverse_);
  if (key.has_value()) {
    auto at_key = ToAtenTensor<kDev>(key->data(), key_shape_, key_strides_,
                                     query_type_, device_index_);
    ApplyRotaryEmbedding(at_key, at_positions, at_cache, key_token_stride_,
                         key_head_stride_, num_kv_heads_, head_size_, rot_dim_,
                         rope_dim_offset_, is_neox_, inverse_);
  }
}

template class Operator<RotaryEmbedding, Device::Type::kCpu, 1>;
template class Operator<RotaryEmbedding, Device::Type::kNvidia, 1>;
template class Operator<RotaryEmbedding, Device::Type::kCambricon, 1>;
template class Operator<RotaryEmbedding, Device::Type::kAscend, 1>;
template class Operator<RotaryEmbedding, Device::Type::kMetax, 1>;
template class Operator<RotaryEmbedding, Device::Type::kMoore, 1>;
template class Operator<RotaryEmbedding, Device::Type::kIluvatar, 1>;
template class Operator<RotaryEmbedding, Device::Type::kHygon, 1>;
template class Operator<RotaryEmbedding, Device::Type::kMars, 1>;

}  // namespace infini::ops
