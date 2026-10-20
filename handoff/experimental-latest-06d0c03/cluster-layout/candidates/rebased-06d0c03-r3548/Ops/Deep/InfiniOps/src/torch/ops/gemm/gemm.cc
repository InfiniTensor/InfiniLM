#include "torch/ops/gemm/gemm.h"

#include "torch/tensor_.h"

namespace infini::ops {

template <Device::Type kDev>
Operator<Gemm, kDev, 2>::Operator(const Tensor a, const Tensor b,
                                  const std::optional<Tensor> c,
                                  std::optional<float> alpha,
                                  std::optional<float> beta,
                                  std::optional<int> trans_a,
                                  std::optional<int> trans_b, Tensor y)
    : Gemm{a, b, c, alpha, beta, trans_a, trans_b, y},
      a_shape_{a.shape()},
      b_shape_{b.shape()},
      y_shape_{y.shape()},
      device_index_{y.device().index()} {}

template <Device::Type kDev>
Operator<Gemm, kDev, 2>::Operator(const Tensor a, const Tensor b, Tensor y)
    : Operator{a,
               b,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               std::nullopt,
               y} {}

template <Device::Type kDev>
void Operator<Gemm, kDev, 2>::operator()(
    const Tensor a, const Tensor b, const std::optional<Tensor> c,
    std::optional<float> alpha, std::optional<float> beta,
    std::optional<int> trans_a, std::optional<int> trans_b, Tensor y) const {
  auto at_a = ToAtenTensor<kDev>(const_cast<void*>(a.data()), a_shape_,
                                 a_strides_, a_type_, device_index_);
  auto at_b = ToAtenTensor<kDev>(const_cast<void*>(b.data()), b_shape_,
                                 b_strides_, b_type_, device_index_);
  auto at_y = ToAtenTensor<kDev>(y.data(), y_shape_, y_strides_, y_type_,
                                 device_index_);

  auto alpha_val = alpha.value_or(alpha_);
  auto beta_val = EffectiveBeta(c, beta);

  if (trans_a.value_or(trans_a_)) {
    at_a = at_a.transpose(-2, -1);
  }

  if (trans_b.value_or(trans_b_)) {
    at_b = at_b.transpose(-2, -1);
  }

  if (alpha_val == 0.0F) {
    at_y.mul_(beta_val);
    return;
  }

  if constexpr (kDev == Device::Type::kCpu || kDev == Device::Type::kNvidia) {
    if (at_a.dim() == 2) {
      at::addmm_out(at_y, at_y, at_a, at_b, beta_val, alpha_val);
    } else {
      at::baddbmm_out(at_y, at_y, at_a, at_b, beta_val, alpha_val);
    }
    return;
  }

  auto product = at::matmul(at_a, at_b);
  if (beta_val == 0.0F) {
    at_y.copy_(product);
    at_y.mul_(alpha_val);
    return;
  }

  at_y.mul_(beta_val);
  at_y.add_(product, alpha_val);
}

template class Operator<Gemm, Device::Type::kCpu, 2>;
template class Operator<Gemm, Device::Type::kNvidia, 2>;
template class Operator<Gemm, Device::Type::kCambricon, 2>;
template class Operator<Gemm, Device::Type::kAscend, 2>;
template class Operator<Gemm, Device::Type::kMetax, 2>;
template class Operator<Gemm, Device::Type::kMoore, 2>;
template class Operator<Gemm, Device::Type::kIluvatar, 2>;
template class Operator<Gemm, Device::Type::kHygon, 2>;
template class Operator<Gemm, Device::Type::kMars, 2>;

}  // namespace infini::ops
