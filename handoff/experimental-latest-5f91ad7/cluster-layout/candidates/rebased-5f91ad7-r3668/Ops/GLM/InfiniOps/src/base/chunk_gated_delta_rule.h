#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>

#include "operator.h"
namespace infini::ops {
class ChunkGatedDeltaRule : public Operator<ChunkGatedDeltaRule> {
 public:
  ChunkGatedDeltaRule(const Tensor q, const Tensor k, const Tensor v,
                      const Tensor g, const Tensor beta, Tensor initial_state,
                      const std::optional<Tensor> cu_seqlens,
                      const std::optional<Tensor> initial_state_indices,
                      const std::optional<Tensor> final_state_indices,
                      Tensor out, std::optional<Tensor> final_state,
                      bool use_qk_l2norm, size_t chunk_size)
      : norm_(use_qk_l2norm),
        chunk_(chunk_size),
        metadata_{q,
                  k,
                  v,
                  g,
                  beta,
                  initial_state,
                  cu_seqlens,
                  initial_state_indices,
                  final_state_indices,
                  out,
                  final_state} {
    const auto& views = metadata_;
    Validate(views, chunk_size);
    device_ = q.device();
    data_type_ = q.dtype();
    gate_type_ = g.dtype();
    B_ = cu_seqlens ? cu_seqlens->numel() - 1 : q.shape()[0];
    T_ = q.shape()[1];
    Hk_ = q.shape()[2];
    Hv_ = v.shape()[2];
    D_ = q.shape()[3];
    pool_ = initial_state.shape()[0];
  }
  virtual void operator()(const Tensor q, const Tensor k, const Tensor v,
                          const Tensor g, const Tensor beta,
                          Tensor initial_state,
                          const std::optional<Tensor> cu_seqlens,
                          const std::optional<Tensor> initial_state_indices,
                          const std::optional<Tensor> final_state_indices,
                          Tensor out, std::optional<Tensor> final_state,
                          bool use_qk_l2norm, size_t chunk_size) const = 0;

 protected:
  using Views = std::array<std::optional<Tensor>, 11>;
  Device device_;
  DataType data_type_, gate_type_;
  bool norm_;
  size_t chunk_, B_, T_, Hk_, Hv_, D_, pool_;
  const Views metadata_;
  static bool floating(DataType dtype) {
    return dtype == DataType::kFloat16 || dtype == DataType::kBFloat16 ||
           dtype == DataType::kFloat32;
  }
  static void Validate(const Views& a, size_t chunk) {
    if (!chunk)
      throw std::invalid_argument("chunk delta chunk_size must be positive");
    const auto &q = *a[0], &k = *a[1], &v = *a[2], &g = *a[3], &beta = *a[4],
               &state = *a[5], &out = *a[9];
    if (!floating(q.dtype()) || !floating(g.dtype()) ||
        beta.dtype() != g.dtype())
      throw std::invalid_argument(
          "chunk delta data/gates support F16/BF16/F32");
    if (q.ndim() != 4 || k.shape() != q.shape() || v.ndim() != 4 ||
        out.shape() != v.shape() || state.ndim() != 4 || g.ndim() != 3 ||
        beta.shape() != g.shape())
      throw std::invalid_argument("chunk delta rank/shape mismatch");
    const auto physical_B = q.shape()[0], T = q.shape()[1], Hk = q.shape()[2],
               Hv = v.shape()[2], D = q.shape()[3], Dv = v.shape()[3];
    size_t B = physical_B;
    if (a[6]) {
      if (physical_B != 1 || a[6]->ndim() != 1 || a[6]->numel() < 2)
        throw std::invalid_argument(
            "chunk delta packed cu_seqlens shape mismatch");
      B = a[6]->numel() - 1;
    }
    if (v.shape()[0] != physical_B || v.shape()[1] != T || !Hk || !Hv ||
        Hv % Hk || !D || !Dv || g.shape() != Tensor::Shape{physical_B, T, Hv})
      throw std::invalid_argument(
          "chunk delta data/gate shape or head grouping mismatch");
    if (a[8].has_value() == a[10].has_value())
      throw std::invalid_argument(
          "chunk delta requires exactly one final destination mode");
    if (state.shape()[1] != Hv || state.shape()[2] != Dv ||
        state.shape()[3] != D || (!a[7] && state.shape()[0] < B) ||
        (a[10] && a[10]->shape() != Tensor::Shape{B, Hv, Dv, D}))
      throw std::invalid_argument(
          "chunk delta state layout must be pool/B,Hv,Dv,Dk");
    for (size_t i = 0; i < a.size(); ++i) {
      if (!a[i]) continue;
      if (a[i]->device() != q.device())
        throw std::invalid_argument("chunk delta device mismatch");
      for (auto stride : a[i]->strides())
        if (stride < 0)
          throw std::invalid_argument(
              "chunk delta negative stride unsupported");
      if (i >= 6 && i <= 8) {
        if (a[i]->shape() != Tensor::Shape{i == 6 ? B + 1 : B} ||
            !a[i]->IsContiguous() ||
            (a[i]->dtype() != DataType::kInt32 &&
             a[i]->dtype() != DataType::kInt64))
          throw std::invalid_argument(
              "chunk delta requires contiguous I32/I64 cu/slot vectors");
      } else if (i != 3 && i != 4 && a[i]->dtype() != q.dtype()) {
        throw std::invalid_argument("chunk delta data dtype mismatch");
      }
      if ((i <= 2 || i == 9) && a[i]->strides()[3] != 1)
        throw std::invalid_argument(
            "chunk delta innermost data dimension must be contiguous");
      if (((i == 5 && a[8]) || i == 9 || i == 10) && a[i]->HasBroadcastDim())
        throw std::invalid_argument(
            "chunk delta broadcast destination unsupported");
    }
    if (T > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        state.shape()[0] >
            static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error(
          "chunk delta token/pool exceeds int64 index extent");
  }
  void CheckInvocation(const Views& views, bool norm, size_t chunk) const {
    if (norm != norm_ || chunk != chunk_)
      throw std::invalid_argument("chunk delta attributes changed after Make");
    for (size_t i = 0; i < views.size(); ++i) {
      if (views[i].has_value() != metadata_[i].has_value() ||
          (views[i] && !std::equal_to<Tensor>{}(*views[i], *metadata_[i])))
        throw std::invalid_argument("chunk delta metadata changed after Make");
      if (views[i] && views[i]->numel() && !views[i]->data())
        throw std::invalid_argument(
            "chunk delta nonempty tensor has null data");
    }
  }
};
}  // namespace infini::ops
