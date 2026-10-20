#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infini::ops::native_w4 {
inline void check(cudaError_t e, const char* operation) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(e));
}
// Restores the previous device; an instance records a device index, never a
// CUDA context or workspace cache. Runtime already owns the active stream.
class DeviceGuard {
 public:
  explicit DeviceGuard(int device) : target_{device} {
    check(cudaGetDevice(&previous_), "W4 cudaGetDevice");
    if (previous_ != target_) check(cudaSetDevice(target_), "W4 cudaSetDevice");
  }
  ~DeviceGuard() {
    if (previous_ != target_) cudaSetDevice(previous_);
  }

 private:
  int previous_{0}, target_{0};
};
inline void no_capture(cudaStream_t stream) {
  cudaStreamCaptureStatus state;
  check(cudaStreamIsCapturing(stream, &state), "W4 capture status");
  if (state != cudaStreamCaptureStatusNone)
    throw std::runtime_error(
        "W4 Marlin capture=false; use explicit recorded host replay");
}
inline void info(int device, int& sms, int& shared) {
  DeviceGuard guard(device);
  int major = 0;
  check(
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device),
      "W4 SM query");
  if (major < 8)
    throw std::runtime_error("W4 Marlin builds SM80+ kernels only");
  check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
        "W4 SM count");
  check(cudaDeviceGetAttribute(&shared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               device),
        "W4 shared memory");
  if (sms <= 0 || shared <= 0)
    throw std::runtime_error("W4 invalid device limits");
}
void prepare_awq_repack(int shared);
void prepare_gptq_repack(int shared, bool permutation);
void launch_awq_repack(const uint32_t* input, uint32_t* output, int k, int n,
                       int sms, int shared, cudaStream_t stream);
void launch_gptq_repack(const uint32_t* input, const uint32_t* perm,
                        uint32_t* output, int k, int n, int sms, int shared,
                        cudaStream_t stream);
}  // namespace infini::ops::native_w4

// Native pointer entry from the fixed upstream marlin.cu; no Tensor wrapper.
namespace vllm {
class ScalarType;
}
namespace marlin {
void marlin_mm(const void* A, const void* B, void* C, void* C_tmp, void* b_bias,
               void* a_s, void* b_s, void* g_s, void* zp, void* g_idx,
               void* perm, void* a_tmp, int prob_m, int prob_n, int prob_k,
               int lda, void* workspace, vllm::ScalarType const& a_type,
               vllm::ScalarType const& b_type, vllm::ScalarType const& c_type,
               vllm::ScalarType const& s_type, bool has_bias,
               bool has_act_order, bool is_k_full, bool has_zp, int num_groups,
               int group_size, int dev, cudaStream_t stream, int thread_k_init,
               int thread_n_init, int sms, bool use_atomic_add,
               bool use_fp32_reduce, bool is_zp_float);
}  // namespace marlin
