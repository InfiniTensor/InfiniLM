#ifdef ENABLE_ATEN
#pragma once
#include "../context/context.hpp"
#include "../tensor.hpp"

#include <ATen/ATen.h>

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
// Hygon (DTK) ships a HIP-flavored libtorch; pulling <c10/cuda/...> under that
// build breaks because c10/cuda/impl/cuda_cmake_macros.h is missing. Hygon
// flash-attn glue lives in a separate hipcc-compiled TU and uses <c10/hip/...>
// directly; this header stays cuda-compat-clean under Hygon.

namespace infinicore::adaptor {
inline at::ScalarType to_at_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::F32:
        return at::kFloat;
    case DataType::F16:
        return at::kHalf;
    case DataType::BF16:
        return at::kBFloat16;
    case DataType::I32:
        return at::kInt;
    case DataType::I64:
        return at::kLong;
    default:
        throw std::runtime_error("Unsupported dtype for ATen");
    }
}

inline at::Device to_at_device(const Device &device) {
    if (device.getType() == Device::Type::NVIDIA) {
        return at::Device(at::kCUDA, device.getIndex());
    } else if (device.getType() == Device::Type::CPU) {
        return at::Device(at::kCPU);
    } else if (device.getType() == Device::Type::QY) {
        return at::Device(at::kCUDA, device.getIndex());
    } else if (device.getType() == Device::Type::HYGON) {
        // DTK's libtorch maps DCU to at::kCUDA via the "MasqueradingAsCUDA"
        // shim (see <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>).
        return at::Device(at::kCUDA, device.getIndex());
    } else {
        throw std::runtime_error("Unsupported device type for ATen");
    }
}

at::Tensor to_aten_tensor(const infinicore::Tensor &t);

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
c10::cuda::CUDAStream get_cuda_stream();
#endif
} // namespace infinicore::adaptor

#endif // ENABLE_ATEN
