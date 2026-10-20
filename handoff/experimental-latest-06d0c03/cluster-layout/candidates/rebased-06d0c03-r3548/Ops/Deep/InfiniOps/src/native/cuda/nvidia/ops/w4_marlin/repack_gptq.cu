#include "launch.h"
#include "gptq_marlin_repack.cu"

namespace infini::ops::native_w4 {
void prepare_gptq_repack(int shared, bool permutation) {
  if (permutation) check(cudaFuncSetAttribute(marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared), "W4 GPTQ repack attributes");
  else check(cudaFuncSetAttribute(marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared), "W4 GPTQ repack attributes");
}
void launch_gptq_repack(const uint32_t* input, const uint32_t* perm, uint32_t* output,
                        int k, int n, int sms, int shared, cudaStream_t stream) {
  if (perm) marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, true, false><<<sms, marlin::repack_threads, shared, stream>>>(input, perm, output, k, n);
  else marlin::gptq_marlin_repack_kernel<marlin::repack_threads, 4, false, false><<<sms, marlin::repack_threads, shared, stream>>>(input, nullptr, output, k, n);
  check(cudaGetLastError(), "W4 GPTQ repack launch");
}
} // namespace infini::ops::native_w4
