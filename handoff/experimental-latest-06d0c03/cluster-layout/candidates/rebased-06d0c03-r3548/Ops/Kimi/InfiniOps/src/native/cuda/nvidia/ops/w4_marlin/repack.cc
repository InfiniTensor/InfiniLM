#include "kernel.h"
#include "launch.h"

namespace infini::ops {
Operator<W4MarlinRepack, Device::Type::kNvidia, 0>::Operator(
    const Tensor qweight, const Tensor perm, const int64_t scheme,
    const int64_t size_k, const int64_t size_n, Tensor out)
    : W4MarlinRepack(qweight, perm, scheme, size_k, size_n, out),
      device_{qweight.device().index()} {
  w4::require(qweight.device().type() == Device::Type::kNvidia,
              "W4 requires one NVIDIA device");
  if (scheme == 1 && perm.numel() && perm.data()) w4::address(perm, 16);
  native_w4::info(device_, sms_, shared_);
  native_w4::DeviceGuard guard(device_);
  if (scheme == 0)
    native_w4::prepare_awq_repack(shared_);
  else
    native_w4::prepare_gptq_repack(shared_, perm.numel() != 0);
}
void Operator<W4MarlinRepack, Device::Type::kNvidia, 0>::operator()(
    const Tensor qweight, const Tensor perm, const int64_t scheme,
    const int64_t size_k, const int64_t size_n, Tensor out) const {
  w4::require(scheme == scheme_ && size_k == k_ && size_n == n_,
              "W4 repack attributes changed");
  w4::address(qweight, 16);
  w4::address(out, 16);
  w4::address(perm, scheme == 1 && perm.numel() ? 16 : 4);
  w4::disjoint(out, qweight);
  w4::disjoint(out, perm);
  native_w4::DeviceGuard guard(device_);
  auto stream = static_cast<cudaStream_t>(stream_);
  native_w4::no_capture(stream);
  auto* output = static_cast<uint32_t*>(out.data());
  auto* input = static_cast<const uint32_t*>(qweight.data());
  if (scheme == 0)
    native_w4::launch_awq_repack(input, output, k_, n_, sms_, shared_, stream);
  else
    native_w4::launch_gptq_repack(
        input,
        perm.numel() ? static_cast<const uint32_t*>(perm.data()) : nullptr,
        output, k_, n_, sms_, shared_, stream);
}
}  // namespace infini::ops
