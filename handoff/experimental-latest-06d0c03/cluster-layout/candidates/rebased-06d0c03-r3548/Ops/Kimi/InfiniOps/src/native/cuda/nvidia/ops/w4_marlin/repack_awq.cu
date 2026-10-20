#include "launch.h"
#include "awq_marlin_repack.cu"

namespace infini::ops::native_w4 {
void prepare_awq_repack(int shared) {
  check(cudaFuncSetAttribute(marlin::awq_marlin_repack_kernel<marlin::repack_threads, 4, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared), "W4 AWQ repack attributes");
}
void launch_awq_repack(const uint32_t* input, uint32_t* output, int k, int n,
                       int sms, int shared, cudaStream_t stream) {
  marlin::awq_marlin_repack_kernel<marlin::repack_threads, 4, false><<<sms, marlin::repack_threads, shared, stream>>>(input, output, k, n);
  check(cudaGetLastError(), "W4 AWQ repack launch");
}
} // namespace infini::ops::native_w4
