#ifndef INFINI_OPS_CAMBRICON_GEMM_CNBLAS_H_
#define INFINI_OPS_CAMBRICON_GEMM_CNBLAS_H_

#include <cassert>
#include <memory>
#include <vector>

// clang-format off
#include <cnnl.h>
#include <cnrt.h>
// clang-format on

#include "base/gemm.h"
#include "native/cambricon/common.h"

namespace infini::ops {

template <>
class Operator<Gemm, Device::Type::kCambricon> : public Gemm {
 public:
  Operator(const Tensor a, const Tensor b, const std::optional<Tensor> c,
           std::optional<float> alpha, std::optional<float> beta,
           std::optional<int> trans_a, std::optional<int> trans_b, Tensor y)
      : Gemm{a, b, c, alpha, beta, trans_a, trans_b, y},
        a_rows_{a.size(-2)},
        a_cols_{a.size(-1)},
        b_rows_{b.size(-2)},
        b_cols_{b.size(-1)},
        y_rows_{y.size(-2)},
        y_cols_{y.size(-1)} {
    assert(!trans_a_ && "`trans_a` is not currently supported");
    assert(!trans_b_ && "`trans_b` is not currently supported");

    cnnlCreate(&cnnl_handle_);

    cnnlCreateTensorDescriptor(&desc_a_);
    cnnlCreateTensorDescriptor(&desc_b_);
    cnnlCreateTensorDescriptor(&desc_y_);

    cnnlCreateMatMulDescriptor(&matmul_desc_);
    cnnlCreateMatMulAlgo(&matmul_algo_);
    cnnlCreateMatMulHeuristicResult(&heuristic_result_);

    int32_t use_stride = 1;
    cnnlSetMatMulDescAttr(matmul_desc_, CNNL_MATMUL_USE_STRIDE, &use_stride,
                          sizeof(int32_t));

    SetupTensorDescriptor(desc_a_, a_strides_, a_type_, a_rows_, a_cols_,
                          batch_count_, batch_stride_a_);
    SetupTensorDescriptor(desc_b_, b_strides_, b_type_, b_rows_, b_cols_,
                          batch_count_, batch_stride_b_);
    SetupTensorDescriptor(desc_y_, y_strides_, y_type_, y_rows_, y_cols_,
                          batch_count_, batch_stride_y_);
    int count = 0;
    cnnlGetBatchMatMulExAlgoHeuristic(cnnl_handle_, matmul_desc_, desc_a_,
                                      desc_b_, desc_y_, NULL, 1,
                                      &heuristic_result_, &count);

    cnrtMalloc(&default_workspace_, workspace_size_in_bytes());
  }

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

  ~Operator() {
    cnrtFree(default_workspace_);
    cnnlDestroyTensorDescriptor(desc_y_);
    cnnlDestroyTensorDescriptor(desc_b_);
    cnnlDestroyTensorDescriptor(desc_a_);
    cnnlDestroyMatMulDescriptor(matmul_desc_);
    cnnlDestroyMatMulAlgo(matmul_algo_);
    cnnlDestroyMatMulHeuristicResult(heuristic_result_);
    cnnlDestroy(cnnl_handle_);
  }

  void operator()(const Tensor a, const Tensor b, const std::optional<Tensor> c,
                  std::optional<float> alpha, std::optional<float> beta,
                  std::optional<int> trans_a, std::optional<int> trans_b,
                  Tensor y) const override {
    const auto& alpha_value{alpha.value_or(alpha_)};
    const auto beta_value{EffectiveBeta(c, beta)};

    cnnlSetQueue(cnnl_handle_, (cnrtQueue_t)stream_);

    auto workspace{workspace_ ? workspace_ : default_workspace_};
    auto workspace_size{workspace_size_in_bytes_ ? workspace_size_in_bytes_
                                                 : workspace_size_in_bytes()};

    cnnlBatchMatMulEx(cnnl_handle_, matmul_desc_, matmul_algo_, &alpha_value,
                      desc_a_, a.data(), desc_b_, b.data(), &beta_value,
                      desc_y_, y.data(), workspace, workspace_size);
  }

  std::size_t workspace_size_in_bytes() const override {
    std::size_t size{0};

    cnnlGetBatchMatMulExHeuristicResult(heuristic_result_, matmul_algo_, &size);

    return size;
  }

 private:
  void SetupTensorDescriptor(cnnlTensorDescriptor_t desc,
                             const Tensor::Strides& strides, DataType dtype,
                             Tensor::Size rows, Tensor::Size cols,
                             Tensor::Size batch, Tensor::Stride batch_stride) {
    cnnlDataType_t cnnl_dtype = cnnl_utils::GetDataType(dtype);

    if (batch > 1) {
      std::vector<int> dims = {static_cast<int>(batch), static_cast<int>(rows),
                               static_cast<int>(cols)};
      std::vector<int> strides_arr = {
          static_cast<int>(batch_stride),
          static_cast<int>(strides[strides.size() - 2]),
          static_cast<int>(strides[strides.size() - 1])};
      cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, cnnl_dtype,
                                dims.size(), dims.data(), strides_arr.data());
    } else {
      std::vector<int> dims = {static_cast<int>(rows), static_cast<int>(cols)};
      std::vector<int> strides_arr = {
          static_cast<int>(strides[strides.size() - 2]),
          static_cast<int>(strides[strides.size() - 1])};
      cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, cnnl_dtype,
                                dims.size(), dims.data(), strides_arr.data());
    }
  }

  cnnlHandle_t cnnl_handle_;

  cnnlTensorDescriptor_t desc_a_;

  cnnlTensorDescriptor_t desc_b_;

  cnnlTensorDescriptor_t desc_y_;

  cnnlMatMulDescriptor_t matmul_desc_;

  cnnlMatMulAlgo_t matmul_algo_;

  cnnlMatMulHeuristicResult_t heuristic_result_;

  Tensor::Size a_rows_, a_cols_;

  Tensor::Size b_rows_, b_cols_;

  Tensor::Size y_rows_, y_cols_;

  // TODO: Remove the following member after default workspace mechanism has
  // been introduced globally.
  void* default_workspace_{nullptr};
};

}  // namespace infini::ops

#endif
