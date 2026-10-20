#ifndef INFINI_OPS_BASE_GEMM_H_
#define INFINI_OPS_BASE_GEMM_H_

#include <algorithm>
#include <cassert>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "operator.h"

namespace infini::ops {

class GemmConfig : public Config {
 public:
  enum class Fp32Precision { DefaultFast, FullFp32 };

  explicit GemmConfig(Fp32Precision precision = Fp32Precision::DefaultFast)
      : fp32_precision_{precision} {
    if (!IsValid(precision)) {
      throw std::invalid_argument("invalid GEMM FP32 precision mode");
    }
  }

  std::unique_ptr<Config> Clone() const override {
    return std::make_unique<GemmConfig>(*this);
  }

  Fp32Precision fp32_precision() const { return fp32_precision_; }

  static Fp32Precision PrecisionFrom(const Config& config) {
    const auto* gemm_config = dynamic_cast<const GemmConfig*>(&config);
    const auto precision = gemm_config ? gemm_config->fp32_precision()
                                       : Fp32Precision::DefaultFast;
    if (!IsValid(precision)) {
      throw std::invalid_argument("invalid GEMM FP32 precision mode");
    }
    return precision;
  }

  static bool SupportsFp32Precision(Device::Type device,
                                    std::size_t implementation_index,
                                    Fp32Precision precision) {
    if (!IsValid(precision)) return false;
    return precision == Fp32Precision::DefaultFast ||
           (device == Device::Type::kNvidia &&
            (implementation_index == 0 || implementation_index == 1));
  }

  static void RequireSupported(const Config& config, Device::Type device) {
    if (!SupportsFp32Precision(device, config.implementation_index(),
                               PrecisionFrom(config))) {
      throw std::invalid_argument(
          "FullFp32 GEMM requires NVIDIA implementation 0 or 1");
    }
  }

  static bool SupportsMetadata(const Config& config, const Tensor& a,
                               const Tensor& b, const Tensor& y) {
    const auto precision = PrecisionFrom(config);
    if (precision == Fp32Precision::DefaultFast) return true;
    return SupportsFp32Precision(a.device().type(),
                                 config.implementation_index(), precision) &&
           a.dtype() == DataType::kFloat32 &&
           b.dtype() == DataType::kFloat32 &&
           y.dtype() == DataType::kFloat32;
  }

  static void RequireSupported(const Config& config, const Tensor& a,
                               const Tensor& b, const Tensor& y) {
    RequireSupported(config, a.device().type());
    if (!SupportsMetadata(config, a, b, y)) {
      throw std::invalid_argument(
          "FullFp32 GEMM requires float32 A, B, and output");
    }
  }

 private:
  static bool IsValid(Fp32Precision precision) {
    return precision == Fp32Precision::DefaultFast ||
           precision == Fp32Precision::FullFp32;
  }

  Fp32Precision fp32_precision_;
};

class Gemm : public Operator<Gemm> {
 public:
  using Operator<Gemm>::Make;

  template <typename... Tail>
  static std::unique_ptr<Operator<Gemm>> Make(const Config& config,
                                               const Tensor a, const Tensor b,
                                               Tail&&... tail) {
    static_assert(sizeof...(Tail) == 1 || sizeof...(Tail) == 6,
                  "GEMM requires short or full argument signature");
    const auto& y = std::get<sizeof...(Tail) - 1>(std::forward_as_tuple(tail...));
    GemmConfig::RequireSupported(config, a, b, y);
    return Operator<Gemm>::Make(config, a, b, std::forward<Tail>(tail)...);
  }

  Gemm(const Tensor a, const Tensor b, const std::optional<Tensor> c,
       std::optional<float> alpha, std::optional<float> beta,
       std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : alpha_{alpha.value_or(1.0)},
        beta_{EffectiveBeta(c, beta)},
        trans_a_{static_cast<bool>(trans_a.value_or(false))},
        trans_b_{static_cast<bool>(trans_b.value_or(false))},
        m_{y.size(-2)},
        n_{y.size(-1)},
        k_{trans_a_ ? a.size(-2) : a.size(-1)},
        a_type_{a.dtype()},
        b_type_{b.dtype()},
        y_type_{y.dtype()},
        a_strides_{a.strides()},
        b_strides_{b.strides()},
        y_strides_{y.strides()},
        lda_{std::max(a.stride(-2), a.stride(-1))},
        ldb_{std::max(b.stride(-2), b.stride(-1))},
        ldy_{std::max(y.stride(-2), y.stride(-1))},
        batch_count_{y.strides().size() > 2 ? y.size(-3) : 1},
        batch_stride_a_{a.strides().size() > 2 ? a.stride(-3) : 0},
        batch_stride_b_{b.strides().size() > 2 ? b.stride(-3) : 0},
        batch_stride_y_{y.strides().size() > 2 ? y.stride(-3) : 0} {
    // TODO: Check constraints.
  }

  Gemm(const Tensor a, const Tensor b, Tensor y)
      : Gemm{a,
             b,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             std::nullopt,
             y} {}

  virtual void operator()(const Tensor a, const Tensor b,
                          const std::optional<Tensor> c,
                          std::optional<float> alpha, std::optional<float> beta,
                          std::optional<int> trans_a,
                          std::optional<int> trans_b, Tensor y) const = 0;

  virtual void operator()(const Tensor a, const Tensor b, Tensor y) const {
    return operator()(a, b, std::nullopt, std::nullopt, std::nullopt,
                      std::nullopt, std::nullopt, y);
  }

  template <typename TensorLike>
  static auto MakeReturnValue(const TensorLike& a, const TensorLike& b) {
    Tensor::Shape y_shape{a.shape()[a.shape().size() - 2],
                          b.shape()[b.shape().size() - 1]};
    return TensorLike::Empty(y_shape, a.dtype(), a.device());
  }

 protected:
  GemmConfig::Fp32Precision fp32_precision(const Tensor& a, const Tensor& b,
                                           const Tensor& y) const {
    const auto precision = config_ptr_
        ? GemmConfig::PrecisionFrom(*config_ptr_)
        : GemmConfig::Fp32Precision::DefaultFast;
    if (precision == GemmConfig::Fp32Precision::FullFp32 &&
        (a.dtype() != DataType::kFloat32 || b.dtype() != DataType::kFloat32 ||
         y.dtype() != DataType::kFloat32)) {
      throw std::invalid_argument(
          "FullFp32 GEMM requires float32 A, B, and output");
    }
    return precision;
  }

  static float EffectiveBeta(const std::optional<Tensor>& c,
                             std::optional<float> beta) {
    static_cast<void>(beta);
    assert(!c && "operator Gemm C input is not supported yet");
    return 0.0F;
  }

  float alpha_{1.0};

  float beta_{1.0};

  bool trans_a_{false};

  bool trans_b_{false};

  Tensor::Size m_{0};

  Tensor::Size n_{0};

  Tensor::Size k_{0};

  const DataType a_type_;

  const DataType b_type_;

  const DataType y_type_;

  Tensor::Strides a_strides_;

  Tensor::Strides b_strides_;

  Tensor::Strides y_strides_;

  Tensor::Stride lda_{0};

  Tensor::Stride ldb_{0};

  Tensor::Stride ldy_{0};

  Tensor::Size batch_count_{1};

  Tensor::Stride batch_stride_a_{0};

  Tensor::Stride batch_stride_b_{0};

  Tensor::Stride batch_stride_y_{0};
};

template <>
struct CacheKeyBuilder<Gemm> {
  template <typename... Tail>
  detail::CacheKey operator()(const Config& config, const Tensor a,
                              const Tensor b, const Tail&... tail) const {
    static_assert(sizeof...(Tail) == 1 || sizeof...(Tail) == 6,
                  "GEMM requires short or full argument signature");
    const auto& y = std::get<sizeof...(Tail) - 1>(std::forward_as_tuple(tail...));
    GemmConfig::RequireSupported(config, a, b, y);
    return detail::CacheKey::Build(
        config.implementation_index(),
        static_cast<int>(GemmConfig::PrecisionFrom(config)), a, b, tail...);
  }
};

}  // namespace infini::ops

#endif
