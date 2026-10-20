#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/numeric_types.h>

#include <cassert>
#include <cstdint>
#include <cute/tensor.hpp>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <type_traits>

#include "native/cuda/nvidia/ops/cutlass_scaled_mm/cutlass.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status{cudaGetDevice(&previous_device_)};
    assert(status == cudaSuccess &&
           "`CutlassScaledMm` failed to query the current CUDA device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`CutlassScaledMm` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status{cudaSetDevice(previous_device_)};
      assert(status == cudaSuccess &&
             "`CutlassScaledMm` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

template <typename ElementOutput, bool kPerToken, bool kPerChannel>
struct GemmDefinition {
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
  using OutputThreadMap =
      cutlass::epilogue::threadblock::OutputTileThreadLayout<
          ThreadblockShape, WarpShape, float, 4, 1>;
  using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
  using Scalar = cutlass::epilogue::threadblock::VisitorScalarBroadcast<float>;
  using PerToken = cutlass::epilogue::threadblock::VisitorColBroadcast<
      OutputThreadMap, float,
      cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;
  using PerChannel = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputThreadMap, float,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;
  using ScaleA = std::conditional_t<kPerToken, PerToken, Scalar>;
  using ScaleB = std::conditional_t<kPerChannel, PerChannel, Scalar>;
  using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
      OutputThreadMap, ElementOutput,
      cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;
  using ScaleBAccumulator = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::multiplies, float, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using ScaleBTree =
      cutlass::epilogue::threadblock::Sm80EVT<ScaleBAccumulator, ScaleB,
                                              Accumulator>;
  using ScaleAAndBias = cutlass::epilogue::threadblock::VisitorCompute<
      cutlass::homogeneous_multiply_add, ElementOutput, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using ComputeTree =
      cutlass::epilogue::threadblock::Sm80EVT<ScaleAAndBias, ScaleA, ScaleBTree,
                                              Bias>;
  using OutputStride = cute::Stride<int64_t, cute::Int<1>, cute::Int<0>>;
  using Output = cutlass::epilogue::threadblock::VisitorAuxStore<
      OutputThreadMap, ElementOutput,
      cutlass::FloatRoundStyle::round_to_nearest, OutputStride>;
  using Epilogue = cutlass::epilogue::threadblock::Sm80EVT<Output, ComputeTree>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
      int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone,
      16, float, cutlass::layout::RowMajor, 4, int32_t, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75, ThreadblockShape,
      WarpShape, InstructionShape, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 2,
      cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename ElementOutput, bool kPerToken, bool kPerChannel>
void Execute(const Tensor a, const Tensor b, const Tensor scale_a,
             const Tensor scale_b, std::optional<Tensor> bias, Tensor out,
             Tensor::Size m, Tensor::Size n, Tensor::Size k, Tensor::Stride lda,
             Tensor::Stride ldb, Tensor::Stride ldo, void* stream) {
  using Definition = GemmDefinition<ElementOutput, kPerToken, kPerChannel>;
  using Gemm = typename Definition::Gemm;

  const auto* a_ptr{reinterpret_cast<const int8_t*>(a.data())};
  const auto* b_ptr{reinterpret_cast<const int8_t*>(b.data())};
  const auto* scale_a_ptr{reinterpret_cast<const float*>(scale_a.data())};
  const auto* scale_b_ptr{reinterpret_cast<const float*>(scale_b.data())};
  const auto* bias_ptr{
      bias ? reinterpret_cast<const ElementOutput*>(bias->data()) : nullptr};
  auto* out_ptr{reinterpret_cast<ElementOutput*>(out.data())};

  typename Definition::ScaleA::Arguments scale_a_arguments{};
  if constexpr (kPerToken) {
    scale_a_arguments.ptr_col = scale_a_ptr;
  } else {
    scale_a_arguments.scalar_ptrs[0] = scale_a_ptr;
  }

  typename Definition::ScaleB::Arguments scale_b_arguments{};
  if constexpr (kPerChannel) {
    scale_b_arguments.ptr_row = scale_b_ptr;
  } else {
    scale_b_arguments.scalar_ptrs[0] = scale_b_ptr;
  }

  typename Definition::Bias::Arguments bias_arguments{};
  bias_arguments.ptr_row = bias_ptr;

  typename Definition::ScaleBTree::Arguments scale_b_tree_arguments{
      scale_b_arguments, {}, {}};
  typename Definition::ComputeTree::Arguments compute_arguments{
      scale_a_arguments, scale_b_tree_arguments, bias_arguments, {}};
  const typename Definition::OutputStride output_stride{
      static_cast<int64_t>(ldo), cute::Int<1>{}, cute::Int<0>{}};
  typename Definition::Output::Arguments output_arguments{out_ptr,
                                                          output_stride};
  typename Definition::Epilogue::Arguments epilogue_arguments{compute_arguments,
                                                              output_arguments};

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)},
      1,
      epilogue_arguments,
      a_ptr,
      b_ptr,
      nullptr,
      nullptr,
      0,
      0,
      0,
      0,
      static_cast<int64_t>(lda),
      static_cast<int64_t>(ldb),
      static_cast<int64_t>(ldo),
      static_cast<int64_t>(ldo)};

  assert(Gemm::get_workspace_size(arguments) == 0 &&
         "`CutlassScaledMm` unexpectedly requires workspace");

  Gemm gemm;
  auto status{gemm.can_implement(arguments)};
  assert(status == cutlass::Status::kSuccess &&
         "CUTLASS cannot implement the requested `CutlassScaledMm` geometry");

  status = gemm(arguments, nullptr, static_cast<cudaStream_t>(stream));
  assert(status == cutlass::Status::kSuccess &&
         "CUTLASS failed to execute `CutlassScaledMm`");

  const auto launch_status{cudaGetLastError()};
  assert(launch_status == cudaSuccess &&
         "`CutlassScaledMm` kernel launch failed");
}

template <typename ElementOutput>
void DispatchScaleLayout(const Tensor a, const Tensor b, const Tensor scale_a,
                         const Tensor scale_b, std::optional<Tensor> bias,
                         Tensor out, Tensor::Size m, Tensor::Size n,
                         Tensor::Size k, Tensor::Stride lda, Tensor::Stride ldb,
                         Tensor::Stride ldo, Tensor::Size scale_a_size,
                         Tensor::Size scale_b_size, void* stream) {
  const auto per_token{scale_a_size != 1};
  const auto per_channel{scale_b_size != 1};

  if (per_token && per_channel) {
    Execute<ElementOutput, true, true>(a, b, scale_a, scale_b, bias, out, m, n,
                                       k, lda, ldb, ldo, stream);
  } else if (per_token) {
    Execute<ElementOutput, true, false>(a, b, scale_a, scale_b, bias, out, m, n,
                                        k, lda, ldb, ldo, stream);
  } else if (per_channel) {
    Execute<ElementOutput, false, true>(a, b, scale_a, scale_b, bias, out, m, n,
                                        k, lda, ldb, ldo, stream);
  } else {
    Execute<ElementOutput, false, false>(a, b, scale_a, scale_b, bias, out, m,
                                         n, k, lda, ldb, ldo, stream);
  }
}

}  // namespace

Operator<CutlassScaledMm, Device::Type::kNvidia, 0>::Operator(
    const Tensor a, const Tensor b, const Tensor scale_a, const Tensor scale_b,
    std::optional<Tensor> bias, const DataType out_dtype, Tensor out)
    : CutlassScaledMm{a, b, scale_a, scale_b, bias, out_dtype, out},
      device_index_{a.device().index()} {
  cudaDeviceProp properties{};
  const auto status{cudaGetDeviceProperties(&properties, device_index_)};
  assert(status == cudaSuccess &&
         "`CutlassScaledMm` failed to query the CUDA device");
  assert((properties.major > 7 ||
          (properties.major == 7 && properties.minor >= 5)) &&
         "`CutlassScaledMm` requires compute capability 7.5 or newer");
}

void Operator<CutlassScaledMm, Device::Type::kNvidia, 0>::operator()(
    const Tensor a, const Tensor b, const Tensor scale_a, const Tensor scale_b,
    std::optional<Tensor> bias, const DataType out_dtype, Tensor out) const {
  assert(out_dtype == out_dtype_ &&
         "`CutlassScaledMm` output dtype changed after descriptor creation");
  assert(out.dtype() == out_dtype_ &&
         "`CutlassScaledMm` output tensor dtype changed after descriptor "
         "creation");
  assert((!bias || bias->dtype() == out_dtype_) &&
         "`CutlassScaledMm` bias dtype changed after descriptor creation");
  DeviceGuard device_guard{device_index_};

  switch (out_dtype_) {
    case DataType::kFloat16:
      DispatchScaleLayout<cutlass::half_t>(
          a, b, scale_a, scale_b, bias, out, m_, n_, k_, lda_, ldb_, ldo_,
          scale_a_size_, scale_b_size_, stream_);
      return;
    case DataType::kBFloat16:
      DispatchScaleLayout<cutlass::bfloat16_t>(
          a, b, scale_a, scale_b, bias, out, m_, n_, k_, lda_, ldb_, ldo_,
          scale_a_size_, scale_b_size_, stream_);
      return;
    default:
      assert(false && "`CutlassScaledMm` received an unsupported output dtype");
  }
}

}  // namespace infini::ops
