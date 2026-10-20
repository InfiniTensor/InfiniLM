#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>

#include "operator.h"

namespace infini::ops {
class CausalConv1d : public Operator<CausalConv1d> {
 public:
  CausalConv1d(const Tensor qkv, const Tensor weight, Tensor conv_state,
               const std::optional<Tensor> bias,
               const std::optional<Tensor> cu_seqlens,
               const std::optional<Tensor> initial_state_indices,
               const std::optional<Tensor> final_state_indices, Tensor out,
               std::optional<Tensor> final_conv_state)
      : metadata_{qkv,
                  weight,
                  conv_state,
                  bias,
                  cu_seqlens,
                  initial_state_indices,
                  final_state_indices,
                  out,
                  final_conv_state} {
    const auto& views = metadata_;
    Validate(views);
    device_ = qkv.device();
    dtype_ = qkv.dtype();
    tokens_ = qkv.shape()[1];
    channels_ = qkv.shape()[2];
    pool_ = conv_state.shape()[0];
    requests_ = cu_seqlens ? cu_seqlens->numel() - 1 : qkv.shape()[0];
  }
  virtual void operator()(const Tensor qkv, const Tensor weight,
                          Tensor conv_state, const std::optional<Tensor> bias,
                          const std::optional<Tensor> cu_seqlens,
                          const std::optional<Tensor> initial_state_indices,
                          const std::optional<Tensor> final_state_indices,
                          Tensor out,
                          std::optional<Tensor> final_conv_state) const = 0;

 protected:
  using Views = std::array<std::optional<Tensor>, 9>;
  Device device_;
  DataType dtype_;
  size_t tokens_, channels_, pool_, requests_;
  const Views metadata_;
  static void Validate(const Views& v) {
    for (const auto& view : v)
      if (view && view->shape().size() != view->strides().size())
        throw std::invalid_argument("causal conv shape/stride rank mismatch");
    const auto &q = *v[0], &w = *v[1], &state = *v[2], &out = *v[7];
    const auto dtype = q.dtype();
    if (dtype != DataType::kFloat32 && dtype != DataType::kFloat16 &&
        dtype != DataType::kBFloat16)
      throw std::invalid_argument("causal conv supports F16/BF16/F32");
    if (q.ndim() != 3 || w.ndim() != 3 || state.ndim() != 3 ||
        out.shape() != q.shape())
      throw std::invalid_argument("causal conv rank/shape mismatch");
    const size_t channels = q.shape()[2];
    if (w.shape()[0] != channels || w.shape()[1] != 1 || w.shape()[2] == 0 ||
        state.shape()[1] != channels || state.shape()[2] != w.shape()[2] - 1)
      throw std::invalid_argument("causal conv weight/state shape mismatch");
    if (v[6].has_value() == v[8].has_value())
      throw std::invalid_argument(
          "causal conv requires exactly one final state destination mode");
    size_t requests = q.shape()[0];
    if (v[4]) {
      if (q.shape()[0] != 1 || v[4]->ndim() != 1 || v[4]->numel() < 2)
        throw std::invalid_argument(
            "causal conv packed sequence shape mismatch");
      requests = v[4]->numel() - 1;
    }
    if (!v[5] && state.shape()[0] < requests)
      throw std::invalid_argument("causal conv state batch mismatch");
    if (v[8] &&
        v[8]->shape() != Tensor::Shape{requests, channels, state.shape()[2]})
      throw std::invalid_argument("causal conv final state shape mismatch");
    if (v[3] && v[3]->shape() != Tensor::Shape{channels})
      throw std::invalid_argument("causal conv bias shape mismatch");
    for (size_t i = 0; i < v.size(); ++i) {
      if (!v[i]) continue;
      if (v[i]->device() != q.device())
        throw std::invalid_argument("causal conv device mismatch");
      if (i >= 4 && i <= 6) {
        if (v[i]->ndim() != 1 || !v[i]->IsContiguous() ||
            (v[i]->dtype() != DataType::kInt32 &&
             v[i]->dtype() != DataType::kInt64) ||
            (i != 4 && v[i]->numel() != requests))
          throw std::invalid_argument(
              "causal conv requires contiguous I32/I64 sequence/slot indices");
      } else if (v[i]->dtype() != dtype && !(i == 1 && v[i]->dtype() == DataType::kFloat32 && q.device().type() == Device::Type::kHygon)) {
        throw std::invalid_argument("causal conv data dtype mismatch");
      }
      // Preserve ordinary positive strided data and reject broadcast
      // destinations.
      if ((i == 2 && v[6]) || i == 7 || i == 8)
        for (size_t axis = 0; axis < v[i]->ndim(); ++axis)
          if (v[i]->shape()[axis] != 1 && v[i]->strides()[axis] == 0)
            throw std::invalid_argument(
                "causal conv broadcast destination unsupported");
    }
    if (q.shape()[1] >
            static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        state.shape()[0] >
            static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error("causal conv index extent exceeds int64");
  }
};
}  // namespace infini::ops
