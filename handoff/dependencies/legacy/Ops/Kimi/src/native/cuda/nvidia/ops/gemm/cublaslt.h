#ifndef INFINI_OPS_NVIDIA_GEMM_CUBLASLT_H_
#define INFINI_OPS_NVIDIA_GEMM_CUBLASLT_H_

#include <cassert>
#include <cstdint>

// clang-format off
#include "cublasLt.h"
// clang-format on

#include "base/gemm.h"
#include "host_range_profiler.h"
#include "native/cuda/nvidia/blas_utils.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kNvidia, 1> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, const std::optional<Tensor> c,
           std::optional<float> alpha, std::optional<float> beta,
           std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : Gemm{a, b, c, alpha, beta, trans_a, trans_b, y},
        a_is_col_major_{a.stride(-1) == 1},
        b_is_col_major_{b.stride(-1) == 1},
        swap_a_and_b_{y.stride(-1) == 1} {}

  Operator(const Tensor a, const Tensor b, Tensor y)
      : Operator{a,
                 b,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 std::nullopt,
                 y} {}

  using Gemm::operator();

  // TODO: Refactor to move initialization/setup logic to the constructor
  // and cleanup/teardown logic to the destructor, rather than executing
  // everything within the computation step.
  // TODO: Replace the current return value checks with utility functions
  // (e.g., `CheckCublasLt`).
  void operator()(const Tensor a, const Tensor b, const std::optional<Tensor> c,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor y) const override {
    [[maybe_unused]] HostRangeScope host_range_backend_submit{
        HostRangeLayer::kBackendSubmit};

    const auto alpha_value{alpha.value_or(alpha_)};
    const auto beta_value{EffectiveBeta(c, beta)};
    const auto trans_a_value{trans_a.value_or(trans_a_)};
    const auto trans_b_value{trans_b.value_or(trans_b_)};

    const auto op_a{GetOpA(trans_a_value, trans_b_value)};
    const auto op_b{GetOpB(trans_a_value, trans_b_value)};
    const auto matmul_m{static_cast<int64_t>(swap_a_and_b_ ? n_ : m_)};
    const auto matmul_n{static_cast<int64_t>(swap_a_and_b_ ? m_ : n_)};
    const auto matmul_k{static_cast<int64_t>(k_)};

    const auto* a_ptr{swap_a_and_b_ ? b.data() : a.data()};
    const auto* b_ptr{swap_a_and_b_ ? a.data() : b.data()};
    const auto a_dtype{BlasUtils<Device::Type::kNvidia>::GetDataType(
        swap_a_and_b_ ? b.dtype() : a.dtype())};
    const auto b_dtype{BlasUtils<Device::Type::kNvidia>::GetDataType(
        swap_a_and_b_ ? a.dtype() : b.dtype())};
    const auto y_dtype{
        BlasUtils<Device::Type::kNvidia>::GetDataType(y.dtype())};
    const auto a_ld{static_cast<uint64_t>(swap_a_and_b_ ? ldb_ : lda_)};
    const auto b_ld{static_cast<uint64_t>(swap_a_and_b_ ? lda_ : ldb_)};
    const auto y_ld{static_cast<uint64_t>(ldy_)};
    const auto a_batch_stride{static_cast<int64_t>(
        swap_a_and_b_ ? batch_stride_b_ : batch_stride_a_)};
    const auto b_batch_stride{static_cast<int64_t>(
        swap_a_and_b_ ? batch_stride_a_ : batch_stride_b_)};
    const auto y_batch_stride{static_cast<int64_t>(batch_stride_y_)};

    cublasLtMatmulDesc_t op_desc{};
    auto status = cublasLtMatmulDescCreate(
        &op_desc, BlasUtils<Device::Type::kNvidia>::GetComputeType(
                      y.dtype(), fp32_precision(a, b, y)),
        CUDA_R_32F);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt matmul descriptor");

    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt transa attribute");

    status = cublasLtMatmulDescSetAttribute(
        op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt transb attribute");

    cublasLtMatrixLayout_t a_layout{};
    status = cublasLtMatrixLayoutCreate(
        &a_layout, a_dtype, op_a == CUBLAS_OP_N ? matmul_m : matmul_k,
        op_a == CUBLAS_OP_N ? matmul_k : matmul_m, a_ld);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt A layout");

    cublasLtMatrixLayout_t b_layout{};
    status = cublasLtMatrixLayoutCreate(
        &b_layout, b_dtype, op_b == CUBLAS_OP_N ? matmul_k : matmul_n,
        op_b == CUBLAS_OP_N ? matmul_n : matmul_k, b_ld);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt B layout");

    cublasLtMatrixLayout_t y_layout{};
    status = cublasLtMatrixLayoutCreate(&y_layout, y_dtype, matmul_m, matmul_n,
                                        y_ld);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt Y layout");

    if (batch_count_ > 1) {
      SetStridedBatchAttributes(a_layout, a_batch_stride);
      SetStridedBatchAttributes(b_layout, b_batch_stride);
      SetStridedBatchAttributes(y_layout, y_batch_stride);
    }

    cublasLtMatmulPreference_t preference{};
    status = cublasLtMatmulPreferenceCreate(&preference);
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to create cuBLASLt preference");

    const auto workspace_size{workspace_size_in_bytes_};
    status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
        sizeof(workspace_size));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt workspace preference");

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned_results{0};
    status = cublasLtMatmulAlgoGetHeuristic(
        GetHandle(), op_desc, a_layout, b_layout, y_layout, y_layout,
        preference, 1, &heuristic, &returned_results);
    assert(status == CUBLAS_STATUS_SUCCESS && returned_results > 0 &&
           "failed to find a cuBLASLt GEMM algorithm");

    status = cublasLtMatmul(
        GetHandle(), op_desc, GetAlphaPtr(alpha_value), a_ptr, a_layout, b_ptr,
        b_layout, GetBetaPtr(beta_value), y.data(), y_layout, y.data(),
        y_layout, &heuristic.algo, workspace_, workspace_size_in_bytes_,
        static_cast<Runtime<Device::Type::kNvidia>::Stream>(stream_));
    assert(status == CUBLAS_STATUS_SUCCESS && "cuBLASLt GEMM launch failed");

    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(y_layout);
    cublasLtMatrixLayoutDestroy(b_layout);
    cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatmulDescDestroy(op_desc);
  }

 private:
  static cublasLtHandle_t& GetHandle() {
    static thread_local cublasLtHandle_t handle = []() {
      cublasLtHandle_t h{};
      auto status = cublasLtCreate(&h);
      assert(status == CUBLAS_STATUS_SUCCESS &&
             "failed to create cuBLASLt handle");
      return h;
    }();
    return handle;
  }

  const void* GetAlphaPtr(const float& alpha) const { return &alpha; }

  const void* GetBetaPtr(const float& beta) const { return &beta; }

  void SetStridedBatchAttributes(cublasLtMatrixLayout_t layout,
                                 int64_t batch_stride) const {
    const int batch_count{static_cast<int>(batch_count_)};
    auto status = cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count,
        sizeof(batch_count));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt batch count");

    status = cublasLtMatrixLayoutSetAttribute(
        layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride,
        sizeof(batch_stride));
    assert(status == CUBLAS_STATUS_SUCCESS &&
           "failed to set cuBLASLt batch stride");
  }

  cublasOperation_t GetOpA(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (b_is_col_major_ == trans_b) ? CUBLAS_OP_T : CUBLAS_OP_N;
    }
    return (a_is_col_major_ != trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
  }

  cublasOperation_t GetOpB(int trans_a, int trans_b) const {
    if (swap_a_and_b_) {
      return (a_is_col_major_ == trans_a) ? CUBLAS_OP_T : CUBLAS_OP_N;
    }
    return (b_is_col_major_ != trans_b) ? CUBLAS_OP_T : CUBLAS_OP_N;
  }

  bool a_is_col_major_{false};

  bool b_is_col_major_{false};

  bool swap_a_and_b_{false};
};

}  // namespace infini::ops

#endif
