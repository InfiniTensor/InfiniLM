#include "kernel.h"
#include "launch.h"

namespace infini::ops::native_w4 {
// These are the byte permutations from the original marlin_utils.cpp; the
// arithmetic keeps FP16/BF16 bit patterns unchanged.
__device__ int scale_source(int i, bool single) {
  if (!single) return (i / 8) + 8 * (i % 8);
  const int j[8] = {0,1,8,9,16,17,24,25};
  return 2*(i/8) + j[i%8];
}
__global__ void scales_reorder(const uint16_t* source, uint16_t* dest, std::size_t count, bool single) {
  const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int block = single ? 32 : 64;
  dest[i] = source[(i/block)*block + scale_source(i%block,single)];
}
__global__ void zeros_reorder(const uint32_t* source, uint32_t* dest, std::size_t words, int n) {
  const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= words) return;
  const int undo[8] = {0,4,1,5,2,6,3,7};
  const int interleave[8] = {0,2,4,6,1,3,5,7};
  const std::size_t row = i/(n/8), col = i%(n/8);
  uint32_t word = 0;
  for (int lane = 0; lane < 8; ++lane) {
    const int permuted = int(col*8) + interleave[lane];
    const int logical = (permuted/64)*64 + scale_source(permuted%64,false);
    const uint32_t raw = source[row*(n/8) + logical/8];
    word |= ((raw >> (4*undo[logical%8])) & 15) << (4*lane);
  }
  dest[i] = word;
}
} // namespace infini::ops::native_w4

namespace infini::ops {
Operator<W4MarlinParameters, Device::Type::kNvidia, 0>::Operator(
    const Tensor scales, const Tensor qzeros, const int64_t scheme,
    const int64_t size_k, const int64_t size_n, const int64_t group_size,
    Tensor scales_out, Tensor zeros_out)
    : W4MarlinParameters(scales,qzeros,scheme,size_k,size_n,group_size,scales_out,zeros_out), device_{scales.device().index()} {
  w4::require(scales.device().type() == Device::Type::kNvidia, "W4 requires one NVIDIA device");
  int sms, shared; native_w4::info(device_,sms,shared);
}
void Operator<W4MarlinParameters, Device::Type::kNvidia, 0>::operator()(
    const Tensor scales, const Tensor qzeros, const int64_t scheme,
    const int64_t size_k, const int64_t size_n, const int64_t group_size,
    Tensor scales_out, Tensor zeros_out) const {
  w4::require(scheme == scheme_ && size_k == k_ && size_n == n_ && group_size == group_, "W4 parameter attributes changed");
  for (const auto& t : {scales,qzeros,scales_out,zeros_out}) w4::address(t,4);
  for (const auto& t : {scales,qzeros}) { w4::disjoint(scales_out,t); w4::disjoint(zeros_out,t); }
  w4::disjoint(scales_out,zeros_out);
  native_w4::DeviceGuard guard(device_);
  auto stream = static_cast<cudaStream_t>(stream_);
  native_w4::no_capture(stream);
  const bool single = group_size == -1 || group_size >= size_k;
  native_w4::scales_reorder<<<(scales.numel()+255)/256,256,0,stream>>>(
      static_cast<const uint16_t*>(scales.data()),static_cast<uint16_t*>(scales_out.data()),scales.numel(),single);
  native_w4::check(cudaGetLastError(), "W4 scale permutation");
  if (scheme == 0) {
    native_w4::zeros_reorder<<<(qzeros.numel()+255)/256,256,0,stream>>>(
        static_cast<const uint32_t*>(qzeros.data()),static_cast<uint32_t*>(zeros_out.data()),qzeros.numel(),n_);
    native_w4::check(cudaGetLastError(), "W4 AWQ zero permutation");
  }
}
} // namespace infini::ops
