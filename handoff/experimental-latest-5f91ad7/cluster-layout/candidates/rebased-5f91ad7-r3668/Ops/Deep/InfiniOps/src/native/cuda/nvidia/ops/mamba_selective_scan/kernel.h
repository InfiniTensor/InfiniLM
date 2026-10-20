#pragma once
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"
#include "native/cuda/ops/mamba_selective_scan/kernel.h"
namespace infini::ops {
struct MambaScanNvidiaBackend : Runtime<Device::Type::kNvidia> {
  static void CheckLaunch() {
    if (cudaGetLastError() != cudaSuccess)
      throw std::runtime_error("MambaSelectiveScan CUDA launch failed");
  }
};
template <>
class Operator<MambaSelectiveScan, Device::Type::kNvidia>
    : public CudaMambaSelectiveScan<MambaScanNvidiaBackend> {
 public:
  using CudaMambaSelectiveScan<MambaScanNvidiaBackend>::CudaMambaSelectiveScan;
};
}  // namespace infini::ops
