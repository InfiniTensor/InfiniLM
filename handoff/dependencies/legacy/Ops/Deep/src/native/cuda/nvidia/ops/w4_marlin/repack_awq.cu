#include "launch.h"
#include "vendor/awq_repack/cuda/kernel.cuh"

namespace infini::ops::native_w4 {
__global__ void awq_repack(const uint32_t* input, uint32_t* output, int k, int n) {
  marlin::awq_marlin_repack_kernel<marlin::repack_threads, 4, false>(input, output, k, n);
}
void prepare_awq_repack(int shared) {
  check(cudaFuncSetAttribute(awq_repack, cudaFuncAttributeMaxDynamicSharedMemorySize, shared), "W4 AWQ repack attributes");
}
void launch_awq_repack(const uint32_t* input, uint32_t* output, int k, int n,
                       int sms, int shared, cudaStream_t stream) {
  awq_repack<<<sms, marlin::repack_threads, shared, stream>>>(input, output, k, n);
  check(cudaGetLastError(), "W4 AWQ repack launch");
}
} // namespace infini::ops::native_w4
