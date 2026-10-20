#include "launch.h"
#include "vendor/gptq_repack/cuda/kernel.cuh"

namespace infini::ops::native_w4 {
template <bool has_perm>
__global__ void gptq_repack(const uint32_t* input, const uint32_t* perm, uint32_t* output, int k, int n) {
  marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, has_perm, false>(input, perm, output, k, n);
}
void prepare_gptq_repack(int shared, bool permutation) {
  if (permutation) check(cudaFuncSetAttribute(gptq_repack<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared), "W4 GPTQ repack attributes");
  else check(cudaFuncSetAttribute(gptq_repack<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared), "W4 GPTQ repack attributes");
}
void launch_gptq_repack(const uint32_t* input, const uint32_t* perm, uint32_t* output,
                        int k, int n, int sms, int shared, cudaStream_t stream) {
  if (perm) gptq_repack<true><<<sms, marlin::repack_threads, shared, stream>>>(input, perm, output, k, n);
  else gptq_repack<false><<<sms, marlin::repack_threads, shared, stream>>>(input, nullptr, output, k, n);
  check(cudaGetLastError(), "W4 GPTQ repack launch");
}
} // namespace infini::ops::native_w4
