#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>

#include "operator.h"
namespace infini::ops {
class RecurrentGatedDeltaRule : public Operator<RecurrentGatedDeltaRule> {
 public:
  RecurrentGatedDeltaRule(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor g, const Tensor beta,
                          Tensor initial_state,
                          const std::optional<Tensor> initial_state_indices,
                          const std::optional<Tensor> final_state_indices,
                          Tensor out, std::optional<Tensor> final_state,
                          bool use_qk_l2norm)
      : norm_(use_qk_l2norm),
        metadata_{q,
                  k,
                  v,
                  g,
                  beta,
                  initial_state,
                  initial_state_indices,
                  final_state_indices,
                  out,
                  final_state} {
    const auto& views = metadata_;
    Validate(views);
    device_ = q.device();
    data_type_ = q.dtype();
    gate_type_ = g.dtype();
    B_ = q.shape()[0];
    Hk_ = q.shape()[2];
    Hv_ = v.shape()[2];
    D_ = q.shape()[3];
    pool_ = initial_state.shape()[0];
  }
  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor g, const Tensor beta,
                          Tensor initial_state,
                          const std::optional<Tensor> initial_state_indices,
                          const std::optional<Tensor> final_state_indices,
                          Tensor out, std::optional<Tensor> final_state,
                          bool use_qk_l2norm) const = 0;

 protected:
  using Views = std::array<std::optional<Tensor>, 10>;
  Device device_;
  DataType data_type_, gate_type_;
  bool norm_;
  size_t B_, Hk_, Hv_, D_, pool_;
  const Views metadata_;
  static bool floating(DataType dtype) {
    return dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
           dtype == DataType::kFloat32;
  }
  static void Validate(const Views& a) {
    const auto &q = *a[0], &k = *a[1], &v = *a[2], &g = *a[3], &beta = *a[4],
               &state = *a[5], &out = *a[8];
    if (!floating(q.dtype()) || !floating(g.dtype()) ||
        beta.dtype() != g.dtype())
      throw std::invalid_argument(
          "recurrent delta data/gates support F16/BF16/F32");
    if (q.ndim() != 4 || k.shape() != q.shape() || v.ndim() != 4 ||
        out.shape() != v.shape() || state.ndim() != 4 || g.ndim() != 3 ||
        beta.shape() != g.shape())
      throw std::invalid_argument("recurrent delta rank/shape mismatch");
    const auto B = q.shape()[0], Hk = q.shape()[2], Hv = v.shape()[2],
               D = q.shape()[3], Dv = v.shape()[3], T = q.shape()[1];
    if (v.shape()[0] != B || v.shape()[1] != T || !Hk || !Hv || Hv % Hk || !D ||
        !Dv || g.shape() != Tensor::Shape{B, T, Hv})
      throw std::invalid_argument(
          "recurrent delta data/gate shape or head grouping mismatch");
    if (a[7].has_value() == a[9].has_value())
      throw std::invalid_argument(
          "recurrent delta requires exactly one final destination mode");
    if (state.shape()[1] != Hv || state.shape()[2] != Dv ||
        state.shape()[3] != D || (!a[6] && state.shape()[0] < B) ||
        (a[9] && a[9]->shape() != Tensor::Shape{B, Hv, Dv, D}))
      throw std::invalid_argument(
          "recurrent delta state layout must be pool/B,Hv,Dv,Dk");
    for (size_t i = 0; i < a.size(); ++i) {
      if (!a[i]) continue;
      if (a[i]->device() != q.device())
        throw std::invalid_argument("recurrent delta device mismatch");
      for (auto stride : a[i]->strides())
        if (stride < 0)
          throw std::invalid_argument(
              "recurrent delta negative stride unsupported");
      if (i == 6 || i == 7) {
        if (a[i]->shape() != Tensor::Shape{B} || !a[i]->IsContiguous() ||
            (a[i]->dtype() != DataType::kInt32 &&
             a[i]->dtype() != DataType::kInt64))
          throw std::invalid_argument(
              "recurrent delta indices require contiguous I32/I64 B-vector");
      } else if (i != 3 && i != 4 && a[i]->dtype() != q.dtype()) {
        throw std::invalid_argument("recurrent delta data dtype mismatch");
      }
      if ((i <= 2 || i == 8) && a[i]->strides()[3] != 1)
        throw std::invalid_argument(
            "recurrent delta innermost data dimension must be contiguous");
      if (((i == 5 && a[7]) || i == 8 || i == 9) && a[i]->HasBroadcastDim())
        throw std::invalid_argument(
            "recurrent delta broadcast destination unsupported");
    }
    if (state.shape()[0] >
        static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error(
          "recurrent delta pool exceeds int64 index extent");
  }
  void CheckInvocation(const Views& views, bool norm) const {
    if (norm != norm_)
      throw std::invalid_argument(
          "recurrent delta normalization changed after Make");
    for (size_t i = 0; i < views.size(); ++i) {
      if (views[i].has_value() != metadata_[i].has_value() ||
          (views[i] && !std::equal_to<Tensor>{}(*views[i], *metadata_[i])))
        throw std::invalid_argument(
            "recurrent delta metadata changed after Make");
      if (views[i] && views[i]->numel() && !views[i]->data())
        throw std::invalid_argument(
            "recurrent delta nonempty tensor has null data");
    }
  }
};
}  // namespace infini::ops
