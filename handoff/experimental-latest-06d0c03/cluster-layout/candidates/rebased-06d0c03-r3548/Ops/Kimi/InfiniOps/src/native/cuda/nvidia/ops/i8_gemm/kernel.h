#pragma once
#include <array>
#include <climits>
#include <limits>

#include "base/i8_gemm.h"
#include "native/cuda/nvidia/ops/binary_view_capture.h"
#include "native/cuda/nvidia/ops/gemm/handle_owner.h"
namespace infini::ops::nvidia_i8_gemm_detail {
struct Layout {
  std::size_t m, n;
  std::int64_t cr, cc, ar, br, sar, sbr, biasr, biasc;
};
void Epilogue(void*, const std::int32_t*, const float*, const float*,
              const void*, DataType, Layout, cudaStream_t);
inline void Check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("I8Gemm cublas error " +
                             std::to_string(int(status)));
}
}  // namespace infini::ops::nvidia_i8_gemm_detail
namespace infini::ops {
template <>
class Operator<I8Gemm, Device::Type::kNvidia, 0> : public I8Gemm {
 public:
  static constexpr std::size_t kBlasWorkspace = 4 * 1024 * 1024;
  static bool supports(Tensor c, const Tensor a, const Tensor as,
                       const Tensor b, const Tensor bs,
                       const std::optional<Tensor> bias) {
    using namespace nvidia_binary_view;
    if (c.device().type() != Device::Type::kNvidia || c.ndim() != 2 ||
        a.ndim() != 2 || b.ndim() != 2 || a.dtype() != DataType::kInt8 ||
        b.dtype() != DataType::kInt8 ||
        (c.dtype() != DataType::kFloat16 && c.dtype() != DataType::kBFloat16 &&
         c.dtype() != DataType::kFloat32) ||
        as.dtype() != DataType::kFloat32 || bs.dtype() != DataType::kFloat32 ||
        c.device() != a.device() || c.device() != b.device() ||
        c.device() != as.device() || c.device() != bs.device())
      return false;
    const auto m = c.shape()[0], n = c.shape()[1], k = a.shape()[1];
    if (!m || !n || !k || m > INT_MAX || n > INT_MAX || k > INT_MAX || n % 4 ||
        k % 4 || a.shape()[0] != m || b.shape() != Tensor::Shape{k, n} ||
        as.shape() != Tensor::Shape{m, 1} ||
        bs.shape() != Tensor::Shape{n, 1} || a.strides().size() != 2 ||
        b.strides().size() != 2 || as.strides().size() != 2 ||
        bs.strides().size() != 2 || a.stride(1) != 1 || b.stride(0) != 1 ||
        a.stride(0) < std::int64_t(k) || b.stride(1) < std::int64_t(k) ||
        a.stride(0) > INT_MAX || b.stride(1) > INT_MAX || a.stride(0) % 4 ||
        b.stride(1) % 4 || !NonOverlappingOutput(c) || as.stride(0) < 1 ||
        bs.stride(0) < 1)
      return false;
    // Worst-case signed INT8 dot cannot overflow int32 under this bound.
    if (k > std::size_t(INT_MAX) / 16384 ||
        m > std::numeric_limits<std::size_t>::max() / n / sizeof(std::int32_t))
      return false;
    int device = -1, major = 0, minor = 0;
    if (cudaGetDevice(&device) != cudaSuccess || device != c.device().index() ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess ||
        major * 10 + minor < 75 ||
        (c.dtype() == DataType::kBFloat16 && major < 8))
      return false;
    if (bias && (bias->device() != c.device() || bias->dtype() != c.dtype() ||
                 bias->shape() != c.shape()))
      return false;
    std::size_t cb = 0;
    if (!SpanBytes(c, &cb)) return false;
    const auto clear = [&](Tensor input) {
      std::size_t bytes = 0;
      return SpanBytes(input, &bytes) && !Overlaps(c, cb, input, bytes) &&
             (!bytes || reinterpret_cast<std::uintptr_t>(input.data()) %
                                input.element_size() ==
                            0);
    };
    if (!clear(a) || !clear(b) || !clear(as) || !clear(bs) ||
        (bias && !clear(*bias)))
      return false;
    return reinterpret_cast<std::uintptr_t>(a.data()) % 4 == 0 &&
           reinterpret_cast<std::uintptr_t>(b.data()) % 4 == 0 &&
           reinterpret_cast<std::uintptr_t>(c.data()) % c.element_size() == 0;
  }
  Operator(Tensor c, const Tensor a, const Tensor as, const Tensor b,
           const Tensor bs, const std::optional<Tensor> bias)
      : I8Gemm(c, a, as, b, bs, bias), dtype_(c.dtype()) {
    const auto m = c.shape()[0], n = c.shape()[1];
    const auto bytes = m * n * sizeof(std::int32_t);
    if (bytes > std::numeric_limits<std::size_t>::max() - 255 - kBlasWorkspace)
      throw std::invalid_argument("I8Gemm workspace size overflow");
    scratch_offset_ = (bytes + 255) & ~std::size_t(255);
    owner_ = NvidiaGemmHandle::Acquire(c.device().index());
    layout_ = {m,
               n,
               c.stride(0),
               c.stride(1),
               a.stride(0),
               b.stride(1),
               as.stride(0),
               bs.stride(0),
               bias ? bias->stride(0) : 0,
               bias ? bias->stride(1) : 0};
  }
  std::size_t workspace_size_in_bytes() const override {
    return scratch_offset_ + kBlasWorkspace;
  }
  bool capture_safe() const override { return true; }
  void operator()(Tensor c, const Tensor a, const Tensor as, const Tensor b,
                  const Tensor bs,
                  const std::optional<Tensor> bias) const override {
    if (!supports(c, a, as, b, bs, bias))
      throw std::invalid_argument("unsupported I8Gemm operands");
    const std::array<Tensor, 5> tensors{c, a, as, b, bs};
    if (!workspace_ || workspace_size_in_bytes_ < workspace_size_in_bytes() ||
        reinterpret_cast<std::uintptr_t>(workspace_) % 256)
      throw std::invalid_argument("I8Gemm requires aligned caller workspace");
    for (const auto& t : tensors)
      if (WorkspaceOverlaps(t))
        throw std::invalid_argument("I8Gemm workspace aliases tensor");
    if (bias && WorkspaceOverlaps(*bias))
      throw std::invalid_argument("I8Gemm workspace aliases bias");
    const auto handle = owner_->get();
    const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
    using nvidia_i8_gemm_detail::Check;
    Check(cublasSetStream(handle, stream));
    Check(cublasSetWorkspace(handle,
                             static_cast<char*>(workspace_) + scratch_offset_,
                             kBlasWorkspace));
    Check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    Check(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    const std::int32_t alpha = 1, beta = 0;
    // Column-major C^T = B^T * A^T. B's [K,N] view stores each
    // original [N,K] weight row contiguously, hence OP_T for B.
    Check(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, int(layout_.n), int(layout_.m),
        int(a.shape()[1]), &alpha, b.data(), CUDA_R_8I, int(layout_.br),
        a.data(), CUDA_R_8I, int(layout_.ar), &beta, workspace_, CUDA_R_32I,
        int(layout_.n), CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
    nvidia_i8_gemm_detail::Epilogue(
        c.data(), static_cast<const std::int32_t*>(workspace_),
        static_cast<const float*>(as.data()),
        static_cast<const float*>(bs.data()), bias ? bias->data() : nullptr,
        dtype_, layout_, stream);
  }

 private:
  bool WorkspaceOverlaps(Tensor t) const {
    std::size_t bytes = 0;
    if (!nvidia_binary_view::SpanBytes(t, &bytes)) return true;
    const auto a = reinterpret_cast<std::uintptr_t>(workspace_),
               b = reinterpret_cast<std::uintptr_t>(t.data());
    return bytes &&
           (a <= b ? b - a < workspace_size_in_bytes() : a - b < bytes);
  }
  DataType dtype_;
  std::size_t scratch_offset_;
  nvidia_i8_gemm_detail::Layout layout_{};
  std::shared_ptr<NvidiaGemmHandle> owner_;
};
}  // namespace infini::ops
