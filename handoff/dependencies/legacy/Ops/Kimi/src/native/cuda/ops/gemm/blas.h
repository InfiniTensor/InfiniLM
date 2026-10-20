#ifndef INFINI_OPS_CUDA_GEMM_BLAS_H_
#define INFINI_OPS_CUDA_GEMM_BLAS_H_

#include <utility>
#include <stdexcept>
#include <string>

#include "base/gemm.h"
#include "native/cuda/blas_utils.h"

namespace infini::ops {

template <typename Backend>
class BlasGemm : public Gemm {
 public:
  BlasGemm(const Tensor a, const Tensor b, const std::optional<Tensor> c,
           std::optional<float> alpha, std::optional<float> beta,
           std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : Gemm{a, b, c, alpha, beta, trans_a, trans_b, y},
        a_is_col_major_{a.stride(-1) == 1},
        b_is_col_major_{b.stride(-1) == 1},
        swap_a_and_b_{y.stride(-1) == 1} {
    // TODO: Check constraints.
  }

  BlasGemm(const Tensor a, const Tensor b, Tensor y)
      : BlasGemm{a,
                 b,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 y} {}

  using Gemm::operator();

  void operator()(const Tensor a, const Tensor b, const std::optional<Tensor> c,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor y) const override {
    const auto handle = GetHandle();
    CheckStatus(Backend::BlasSetStream(
                    handle, static_cast<typename Backend::Stream>(stream_)),
                "setting the GEMM stream");

    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto beta_value{EffectiveBeta(c, beta)};

    const auto& trans_a_value{trans_a.value_or(trans_a_)};
    const auto& trans_b_value{trans_b.value_or(trans_b_)};
    auto op_a{GetOpA(trans_a_value, trans_b_value)};
    auto op_b{GetOpB(trans_a_value, trans_b_value)};
    const void* alpha_ptr{GetAlphaPtr(alpha_value, y.dtype())};
    const void* beta_ptr{GetBetaPtr(beta_value, y.dtype())};
    auto compute_type = BlasUtils<Backend::kDeviceType>::GetComputeType(y.dtype());
    if constexpr (Backend::kDeviceType == Device::Type::kNvidia) {
      compute_type = BlasUtils<Backend::kDeviceType>::GetComputeType(
          y.dtype(), fp32_precision(a, b, y));
    }

    CheckStatus(Backend::BlasGemmStridedBatchedEx(
        handle, op_a, op_b, swap_a_and_b_ ? n_ : m_,
        swap_a_and_b_ ? m_ : n_, k_, alpha_ptr,
        swap_a_and_b_ ? b.data() : a.data(),
        BlasUtils<Backend::kDeviceType>::GetDataType(swap_a_and_b_ ? b.dtype()
                                                                   : a.dtype()),
        swap_a_and_b_ ? ldb_ : lda_,
        swap_a_and_b_ ? batch_stride_b_ : batch_stride_a_,
        swap_a_and_b_ ? a.data() : b.data(),
        BlasUtils<Backend::kDeviceType>::GetDataType(swap_a_and_b_ ? a.dtype()
                                                                   : b.dtype()),
        swap_a_and_b_ ? lda_ : ldb_,
        swap_a_and_b_ ? batch_stride_a_ : batch_stride_b_, beta_ptr, y.data(),
        BlasUtils<Backend::kDeviceType>::GetDataType(y.dtype()), ldy_,
        batch_stride_y_, batch_count_,
        compute_type,
        Backend::BLAS_GEMM_DEFAULT), "submitting GEMM");
  }

 protected:
  virtual const void* GetAlphaPtr(const float& alpha, DataType) const {
    return &alpha;
  }

  virtual const void* GetBetaPtr(const float& beta, DataType) const {
    return &beta;
  }

 private:
  auto GetOpA(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (b_is_col_major_ == trans_b) ? Backend::BLAS_OP_T
                                          : Backend::BLAS_OP_N;
    }
    return (a_is_col_major_ != trans_a) ? Backend::BLAS_OP_T
                                        : Backend::BLAS_OP_N;
  }

  auto GetOpB(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (a_is_col_major_ == trans_a) ? Backend::BLAS_OP_T
                                          : Backend::BLAS_OP_N;
    }
    return (b_is_col_major_ != trans_b) ? Backend::BLAS_OP_T
                                        : Backend::BLAS_OP_N;
  }

 protected:
  virtual typename Backend::BlasHandle GetHandle() const {
    thread_local typename Backend::BlasHandle handle = []() {
      typename Backend::BlasHandle h{};
      const auto status = Backend::BlasCreate(&h);
      if (status != 0 || h == nullptr) {
        if (h != nullptr) Backend::BlasDestroy(h);
        throw std::runtime_error("creating GEMM BLAS handle failed: " +
                                 std::to_string(static_cast<int>(status)));
      }
      return h;
    }();
    return handle;
  }

 private:
  template <typename Status>
  static void CheckStatus(Status status, const char* operation) {
    if (status != 0) {
      throw std::runtime_error(std::string(operation) + " failed: " +
                               std::to_string(static_cast<int>(status)));
    }
  }

  bool a_is_col_major_{false};

  bool b_is_col_major_{false};

  bool swap_a_and_b_{false};
};

}  // namespace infini::ops

#endif
