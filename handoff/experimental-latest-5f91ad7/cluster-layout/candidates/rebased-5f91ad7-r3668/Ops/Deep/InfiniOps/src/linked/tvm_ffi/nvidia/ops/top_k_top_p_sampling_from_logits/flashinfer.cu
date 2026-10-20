#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

#include "data_type.h"
#include "dispatcher.h"
#include "linked/tvm_ffi/nvidia/ops/top_k_top_p_sampling_from_logits/flashinfer.h"
#include "native/cpu/caster_.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/nvidia/runtime_.h"

extern "C" {
int __tvm_ffi_softmax(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);
int __tvm_ffi_top_k_mask_logits(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);
int __tvm_ffi_top_p_sampling_from_probs(void*, const TVMFFIAny*, int32_t,
                                        TVMFFIAny*);
int __tvm_ffi_top_k_top_p_sampling_from_probs(void*, const TVMFFIAny*, int32_t,
                                              TVMFFIAny*);
}

namespace infini::ops {
namespace {

using OptionalTensorView = tvm::ffi::Optional<tvm::ffi::TensorView>;

constexpr std::size_t kScratchBytes = 1024 * 1024;
constexpr std::size_t kAlignment = 256;
constexpr unsigned int kThreads = 256;

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

void CheckCuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string{operation} + ": " +
                             cudaGetErrorString(status));
  }
}

void ReportCuda(cudaError_t status, const char* operation) noexcept {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "[InfiniOps] %s: %s\n", operation,
                 cudaGetErrorString(status));
  }
}

void CheckTvmFfi(int status, const char* operation) {
  if (status != 0) {
    throw std::runtime_error(std::string{operation} + " (status " +
                             std::to_string(status) + ")");
  }
}

void ReportTvmFfi(int status, const char* operation) noexcept {
  if (status != 0) {
    std::fprintf(stderr, "[InfiniOps] %s (status %d)\n", operation, status);
  }
}

std::size_t Align(std::size_t value) {
  return (value + kAlignment - 1) & ~(kAlignment - 1);
}

std::size_t AddWorkspaceRegion(std::size_t* offset, std::size_t size) {
  *offset = Align(*offset);
  const auto result = *offset;
  *offset += size;
  return result;
}

struct WorkspaceLayout {
  explicit WorkspaceLayout(std::size_t matrix_elements,
                           std::size_t batch_size) {
    matrix_a = AddWorkspaceRegion(&size, matrix_elements * sizeof(float));
    matrix_b = AddWorkspaceRegion(&size, matrix_elements * sizeof(float));
    top_k = AddWorkspaceRegion(&size, batch_size * sizeof(int64_t));
    top_p = AddWorkspaceRegion(&size, batch_size * sizeof(float));
    valid = AddWorkspaceRegion(&size, batch_size * sizeof(uint8_t));
    indices = AddWorkspaceRegion(&size, batch_size * sizeof(int64_t));
    scratch = AddWorkspaceRegion(&size, kScratchBytes);
    size = Align(size);
  }

  std::size_t matrix_a{0};
  std::size_t matrix_b{0};
  std::size_t top_k{0};
  std::size_t top_p{0};
  std::size_t valid{0};
  std::size_t indices{0};
  std::size_t scratch{0};
  std::size_t size{0};
};

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    CheckCuda(status,
              "FlashInferSampling failed to query the current CUDA device");
    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      CheckCuda(status,
                "FlashInferSampling failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (!restore_) return;
    const auto status = cudaSetDevice(previous_device_);
    ReportCuda(status,
               "FlashInferSampling failed to restore the CUDA device");
  }

 private:
  int previous_device_{0};
  bool restore_{false};
};

class StreamGuard {
 public:
  StreamGuard(int device_index, cudaStream_t stream)
      : device_index_{device_index} {
    const auto status =
        TVMFFIEnvSetStream(kDLCUDA, device_index, stream, &previous_stream_);
    CheckTvmFfi(status,
                "FlashInferSampling failed to set the TVM FFI CUDA stream");
  }

  ~StreamGuard() {
    const auto status =
        TVMFFIEnvSetStream(kDLCUDA, device_index_, previous_stream_, nullptr);
    ReportTvmFfi(status,
                 "FlashInferSampling failed to restore the TVM FFI CUDA stream");
  }

 private:
  int device_index_{0};
  TVMFFIStreamHandle previous_stream_{nullptr};
};

class Workspace {
 public:
  Workspace(void* external, std::size_t available, std::size_t required,
            cudaStream_t stream)
      : data_{external}, stream_{stream} {
    if (external != nullptr) {
      Require(available >= required,
              "FlashInferSampling received insufficient workspace");
      return;
    }

    CheckCuda(cudaMallocAsync(&data_, required, stream_),
              "FlashInferSampling failed to allocate async workspace");
    owned_ = true;
  }

  ~Workspace() {
    if (owned_) {
      ReportCuda(cudaFreeAsync(data_, stream_),
                 "FlashInferSampling failed to free async workspace");
    }
  }

  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;

  void* data() const { return data_; }

 private:
  void* data_{nullptr};
  cudaStream_t stream_{nullptr};
  bool owned_{false};
};

class StagingRecorder {
 public:
  StagingRecorder(cudaEvent_t event, cudaStream_t stream, bool* recorded)
      : event_{event}, stream_{stream}, recorded_{recorded} {
    *recorded_ = false;
  }

  ~StagingRecorder() {
    if (!active_) return;

    const auto status = cudaEventRecord(event_, stream_);
    if (status == cudaSuccess) {
      *recorded_ = true;
      return;
    }

    ReportCuda(status,
               "FlashInferSampling failed to record staging completion");
    ReportCuda(cudaStreamSynchronize(stream_),
               "FlashInferSampling failed to await staging after an error");
  }

  StagingRecorder(const StagingRecorder&) = delete;
  StagingRecorder& operator=(const StagingRecorder&) = delete;

  void Record() {
    active_ = false;
    const auto status = cudaEventRecord(event_, stream_);
    if (status != cudaSuccess) {
      CheckCuda(
          cudaStreamSynchronize(stream_),
          "FlashInferSampling failed to await staging after a record error");
      CheckCuda(status,
                "FlashInferSampling failed to record staging completion");
    }
    *recorded_ = true;
  }

 private:
  cudaEvent_t event_{nullptr};
  cudaStream_t stream_{nullptr};
  bool* recorded_{nullptr};
  bool active_{true};
};

DLDataType Dtype(DataType dtype) {
  switch (dtype) {
    case DataType::kInt32:
      return {kDLInt, 32, 1};
    case DataType::kInt64:
      return {kDLInt, 64, 1};
    case DataType::kFloat32:
      return {kDLFloat, 32, 1};
    default:
      throw std::invalid_argument(
          "FlashInferSampling received an unsupported dtype");
  }
}

DLTensor MakeTensor(void* data, int device_index, int32_t ndim, int64_t* shape,
                    DLDataType dtype) {
  return {data, {kDLCUDA, device_index}, ndim, dtype, shape, nullptr, 0};
}

template <typename Src>
__global__ void CastLogits(float* dst, const Src* src, std::size_t count) {
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    dst[index] = Caster<Device::Type::kNvidia>::Cast<float>(src[index]);
  }
}

template <typename Src, typename Index>
__global__ void GatherCastLogits(float* dst, const Src* src,
                                 const Index* indices, std::size_t rows,
                                 std::size_t source_rows,
                                 std::size_t vocab_size) {
  const auto count = rows * vocab_size;
  for (auto index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const auto row = index / vocab_size;
    const auto column = index % vocab_size;
    const auto source_index = indices[row];
    if (source_index < 0 ||
        static_cast<std::size_t>(source_index) >= source_rows) {
      asm("trap;");
      return;
    }
    const auto source_row = static_cast<std::size_t>(source_index);
    dst[index] = Caster<Device::Type::kNvidia>::Cast<float>(
        src[source_row * vocab_size + column]);
  }
}

unsigned int Blocks(std::size_t count) {
  const auto blocks = (count + kThreads - 1) / kThreads;
  return static_cast<unsigned int>(std::min<std::size_t>(blocks, 65535));
}

void CallSoftmax(DLTensor* scratch, DLTensor* logits, DLTensor* output) {
  tvm::ffi::Function::InvokeExternC(
      nullptr, __tvm_ffi_softmax, tvm::ffi::TensorView(scratch),
      tvm::ffi::TensorView(logits), tvm::ffi::TensorView(output),
      OptionalTensorView{}, 1.0, false);
}

void CallTopKMask(DLTensor* logits, DLTensor* output, DLTensor* top_k,
                  DLTensor* scratch) {
  tvm::ffi::Function::InvokeExternC(
      nullptr, __tvm_ffi_top_k_mask_logits, tvm::ffi::TensorView(logits),
      tvm::ffi::TensorView(output),
      OptionalTensorView{tvm::ffi::TensorView(top_k)}, int64_t{0},
      tvm::ffi::TensorView(scratch));
}

OptionalTensorView OptionalView(DLTensor* tensor) {
  return tensor == nullptr ? OptionalTensorView{}
                           : OptionalTensorView{tvm::ffi::TensorView(tensor)};
}

void CallTopP(DLTensor* probs, DLTensor* output, DLTensor* valid,
              DLTensor* indices, DLTensor* top_p, bool deterministic,
              uint64_t seed, uint64_t offset) {
  tvm::ffi::Function::InvokeExternC(
      nullptr, __tvm_ffi_top_p_sampling_from_probs, tvm::ffi::TensorView(probs),
      tvm::ffi::TensorView(output), tvm::ffi::TensorView(valid),
      OptionalView(indices), OptionalTensorView{tvm::ffi::TensorView(top_p)},
      1.0, deterministic, OptionalTensorView{}, seed, OptionalTensorView{},
      offset);
}

void CallJoint(DLTensor* probs, DLTensor* output, DLTensor* valid,
               DLTensor* indices, DLTensor* top_k, DLTensor* top_p,
               bool deterministic, uint64_t seed, uint64_t offset) {
  tvm::ffi::Function::InvokeExternC(
      nullptr, __tvm_ffi_top_k_top_p_sampling_from_probs,
      tvm::ffi::TensorView(probs), tvm::ffi::TensorView(output),
      tvm::ffi::TensorView(valid), OptionalView(indices),
      OptionalTensorView{tvm::ffi::TensorView(top_k)}, 0.0,
      OptionalTensorView{tvm::ffi::TensorView(top_p)}, 1.0, deterministic,
      OptionalTensorView{}, seed, OptionalTensorView{}, offset);
}

int64_t ReadTopK(const Tensor tensor, Tensor::Size row) {
  const auto offset = row * tensor.stride(0);
  return tensor.dtype() == DataType::kInt32
             ? static_cast<const int32_t*>(tensor.data())[offset]
             : static_cast<const int64_t*>(tensor.data())[offset];
}

double ReadTopP(const Tensor tensor, Tensor::Size row) {
  const auto offset = row * tensor.stride(0);
  switch (tensor.dtype()) {
    case DataType::kFloat16:
      return Caster<Device::Type::kCpu>::Cast<float>(
          static_cast<const Float16*>(tensor.data())[offset]);
    case DataType::kBFloat16:
      return Caster<Device::Type::kCpu>::Cast<float>(
          static_cast<const BFloat16*>(tensor.data())[offset]);
    case DataType::kFloat32:
      return static_cast<const float*>(tensor.data())[offset];
    case DataType::kFloat64:
      return static_cast<const double*>(tensor.data())[offset];
    default:
      throw std::invalid_argument(
          "FlashInferSampling received an unsupported top-p dtype");
  }
}

float NormalizeTopP(double value) {
  if (!(value > 0.0 && value < 1.0)) return 1.0f;

  const auto converted = static_cast<float>(value);
  if (converted <= 0.0f) {
    return std::numeric_limits<float>::denorm_min();
  }
  if (converted >= 1.0f) {
    return std::nextafter(1.0f, 0.0f);
  }
  return converted;
}

void Validate(const Tensor logits, const Tensor top_k, const Tensor top_p,
              const std::optional<Tensor>& indices,
              const std::string& filter_apply_order, bool check_nan,
              Tensor out) {
  const auto logits_dtype = logits.dtype();
  Require(logits_dtype == DataType::kFloat16 ||
              logits_dtype == DataType::kBFloat16 ||
              logits_dtype == DataType::kFloat32,
          "FlashInferSampling supports float16, bfloat16, or float32 logits");
  Require(logits.device().type() == Device::Type::kNvidia &&
              out.device() == logits.device() && logits.IsContiguous() &&
              out.IsContiguous(),
          "FlashInferSampling requires contiguous NVIDIA logits and output");
  Require((top_k.dtype() == DataType::kInt32 ||
           top_k.dtype() == DataType::kInt64) &&
              top_k.device().type() == Device::Type::kCpu,
          "FlashInferSampling requires host int32 or int64 top-k");
  Require((top_p.dtype() == DataType::kFloat16 ||
           top_p.dtype() == DataType::kBFloat16 ||
           top_p.dtype() == DataType::kFloat32 ||
           top_p.dtype() == DataType::kFloat64) &&
              top_p.device().type() == Device::Type::kCpu,
          "FlashInferSampling requires host floating-point top-p");
  Require(out.dtype() == DataType::kInt32 ||
              out.dtype() == DataType::kInt64,
          "FlashInferSampling requires int32 or int64 output");
  Require(filter_apply_order == "top_k_first" ||
              filter_apply_order == "joint",
          "FlashInferSampling requires top_k_first or joint filter order");
  Require(!check_nan, "FlashInferSampling does not support check_nan");
  if (indices) {
    Require((indices->device() == logits.device() ||
             indices->device().type() == Device::Type::kCpu) &&
                indices->IsContiguous() && indices->dtype() == out.dtype(),
            "FlashInferSampling requires contiguous CPU or NVIDIA indices "
            "matching output");
  } else {
    Require(logits.size(0) == out.size(0),
            "FlashInferSampling requires output batch size to match logits "
            "when indices are absent");
    Require(out.dtype() == DataType::kInt32,
            "FlashInferSampling requires int32 output when indices are "
            "absent");
  }
}

}  // namespace

Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia, 16>::Operator(
    const Tensor logits, const Tensor top_k, const Tensor top_p,
    const std::optional<Tensor> indices, const std::string filter_apply_order,
    const bool deterministic, const bool check_nan,
    const std::optional<int64_t> seed, const std::optional<int64_t> offset,
    Tensor out)
    : TopKTopPSamplingFromLogits(logits, top_k, top_p, indices,
                                 filter_apply_order, deterministic, check_nan,
                                 seed, offset, out),
      workspace_size_{
          WorkspaceLayout(static_cast<std::size_t>(out.size(0)) *
                              static_cast<std::size_t>(logits.size(1)),
                          static_cast<std::size_t>(out.size(0)))
              .size},
      logits_batch_size_{logits.size(0)},
      device_index_{logits.device().index()},
      top_k_dtype_{top_k.dtype()},
      top_p_dtype_{top_p.dtype()},
      out_dtype_{out.dtype()},
      indices_dtype_{indices ? std::optional{indices->dtype()} : std::nullopt},
      indices_device_{indices ? std::optional{indices->device()}
                              : std::nullopt},
      filter_apply_order_{filter_apply_order},
      deterministic_{deterministic} {
  Validate(logits, top_k, top_p, indices, filter_apply_order, check_nan, out);
  Require(vocab_size_ > 0 &&
              vocab_size_ <= static_cast<Tensor::Size>(
                                 std::numeric_limits<int32_t>::max()),
          "FlashInferSampling requires a nonempty int32-sized vocabulary");
  if (batch_size_ == 0) return;

  DeviceGuard guard{device_index_};
  try {
    for (auto& slot : staging_slots_) {
      CheckCuda(
          cudaMallocHost(
              &slot.top_p,
              static_cast<std::size_t>(batch_size_) * sizeof(float)),
          "FlashInferSampling failed to allocate top-p staging");
      CheckCuda(
          cudaMallocHost(
              &slot.top_k,
              static_cast<std::size_t>(batch_size_) * sizeof(int64_t)),
          "FlashInferSampling failed to allocate top-k staging");
      CheckCuda(
          cudaMallocHost(
              &slot.indices,
              static_cast<std::size_t>(batch_size_) * sizeof(int64_t)),
          "FlashInferSampling failed to allocate indices staging");

      cudaEvent_t event{nullptr};
      CheckCuda(cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
                "FlashInferSampling failed to create staging event");
      slot.event = event;
    }
  } catch (...) {
    ReleaseStagingSlots();
    throw;
  }
}

void Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
              16>::ReleaseStagingSlots() noexcept {
  for (auto& slot : staging_slots_) {
    if (slot.event_recorded && slot.event != nullptr) {
      const auto status =
          cudaEventSynchronize(static_cast<cudaEvent_t>(slot.event));
      if (status != cudaSuccess) {
        ReportCuda(status, "FlashInferSampling failed to await staging");
        ReportCuda(cudaDeviceSynchronize(),
                   "FlashInferSampling failed to await device work");
      }
    }
    if (slot.event != nullptr) {
      ReportCuda(cudaEventDestroy(static_cast<cudaEvent_t>(slot.event)),
                 "FlashInferSampling failed to destroy staging event");
    }
    if (slot.indices != nullptr) {
      ReportCuda(cudaFreeHost(slot.indices),
                 "FlashInferSampling failed to free indices staging");
    }
    if (slot.top_k != nullptr) {
      ReportCuda(cudaFreeHost(slot.top_k),
                 "FlashInferSampling failed to free top-k staging");
    }
    if (slot.top_p != nullptr) {
      ReportCuda(cudaFreeHost(slot.top_p),
                 "FlashInferSampling failed to free top-p staging");
    }
    slot = {};
  }
}

Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia, 16>::~Operator() {
  if (batch_size_ == 0) return;

  int previous_device{0};
  auto status = cudaGetDevice(&previous_device);
  if (status != cudaSuccess) {
    ReportCuda(status,
               "FlashInferSampling failed to query the current CUDA device");
    return;
  }

  const bool restore_device = previous_device != device_index_;
  if (restore_device) {
    status = cudaSetDevice(device_index_);
    if (status != cudaSuccess) {
      ReportCuda(status,
                 "FlashInferSampling failed to select the input CUDA device");
      return;
    }
  }

  ReleaseStagingSlots();
  if (restore_device) {
    ReportCuda(cudaSetDevice(previous_device),
               "FlashInferSampling failed to restore the CUDA device");
  }
}

std::size_t Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
                     16>::workspace_size_in_bytes() const {
  return workspace_size_;
}

void Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia,
              16>::operator()(const Tensor logits, const Tensor top_k,
                              const Tensor top_p,
                              const std::optional<Tensor> indices,
                              const std::string filter_apply_order,
                              const bool deterministic, const bool check_nan,
                              const std::optional<int64_t> seed,
                              const std::optional<int64_t> offset,
                              Tensor out) const {
  Require(logits.ndim() == 2 && logits.size(0) == logits_batch_size_ &&
              logits.size(1) == vocab_size_ && logits.dtype() == dtype_ &&
              logits.device().type() == Device::Type::kNvidia &&
              logits.device().index() == device_index_ && top_k.ndim() == 1 &&
              top_k.size(0) == batch_size_ &&
              top_k.dtype() == top_k_dtype_ &&
              top_k.device().type() == Device::Type::kCpu &&
              top_p.ndim() == 1 && top_p.size(0) == batch_size_ &&
              top_p.dtype() == top_p_dtype_ &&
              top_p.device().type() == Device::Type::kCpu && out.ndim() == 1 &&
              out.size(0) == batch_size_ && out.dtype() == out_dtype_ &&
              out.device() == logits.device() &&
              indices.has_value() == indices_dtype_.has_value() &&
              filter_apply_order == filter_apply_order_ &&
              deterministic == deterministic_,
          "FlashInferSampling call metadata changed after descriptor creation");
  if (indices) {
    Require(indices->ndim() == 1 && indices->size(0) == batch_size_ &&
                indices->dtype() == *indices_dtype_ &&
                indices->device() == *indices_device_,
            "FlashInferSampling indices metadata changed after descriptor "
            "creation");
  }
  Require(!offset || *offset >= 0,
          "FlashInferSampling requires a nonnegative offset");
  Validate(logits, top_k, top_p, indices, filter_apply_order, check_nan, out);
  if (batch_size_ == 0) return;

  DeviceGuard device_guard{device_index_};
  const auto stream = static_cast<cudaStream_t>(stream_);
  StreamGuard stream_guard{device_index_, stream};
  std::lock_guard lock{mutex_};

  auto slot_index = next_staging_slot_;
  auto* slot = &staging_slots_[slot_index];
  auto status = cudaSuccess;
  if (slot->event_recorded) {
    status = cudaEventQuery(static_cast<cudaEvent_t>(slot->event));
    if (status == cudaErrorNotReady) {
      const auto other_index = (slot_index + 1) % staging_slots_.size();
      auto* other = &staging_slots_[other_index];
      auto other_status =
          other->event_recorded
              ? cudaEventQuery(static_cast<cudaEvent_t>(other->event))
              : cudaSuccess;
      if (other_status == cudaSuccess) {
        slot_index = other_index;
        slot = other;
      } else if (other_status == cudaErrorNotReady) {
        CheckCuda(cudaEventSynchronize(static_cast<cudaEvent_t>(slot->event)),
                  "FlashInferSampling failed to await staging slot");
      } else {
        CheckCuda(other_status,
                  "FlashInferSampling failed to query staging event");
      }
    } else if (status != cudaSuccess) {
      CheckCuda(status, "FlashInferSampling failed to query staging event");
    }
  }
  next_staging_slot_ = (slot_index + 1) % staging_slots_.size();

  const auto matrix_elements = static_cast<std::size_t>(batch_size_) *
                               static_cast<std::size_t>(logits.size(1));
  const WorkspaceLayout layout{matrix_elements,
                               static_cast<std::size_t>(batch_size_)};
  Workspace workspace_owner{workspace_, workspace_size_in_bytes_, layout.size,
                            stream};
  auto* workspace = static_cast<uint8_t*>(workspace_owner.data());
  auto* matrix_a = reinterpret_cast<float*>(workspace + layout.matrix_a);
  auto* matrix_b = reinterpret_cast<float*>(workspace + layout.matrix_b);
  auto* top_k_device = workspace + layout.top_k;
  auto* top_p_device = reinterpret_cast<float*>(workspace + layout.top_p);
  auto* valid_device = workspace + layout.valid;
  auto* scratch_device = workspace + layout.scratch;
  auto* indices_device = workspace + layout.indices;

  for (Tensor::Size row = 0; row < batch_size_; ++row) {
    slot->top_p[static_cast<std::size_t>(row)] =
        NormalizeTopP(ReadTopP(top_p, row));
  }

  const bool top_k_is_int64 =
      filter_apply_order == "joint" && out.dtype() == DataType::kInt64;
  std::size_t top_k_bytes{0};
  if (top_k_is_int64) {
    for (Tensor::Size row = 0; row < batch_size_; ++row) {
      const auto value = ReadTopK(top_k, row);
      slot->top_k[static_cast<std::size_t>(row)] =
          value > 0 && value <= static_cast<int64_t>(vocab_size_)
              ? value
              : static_cast<int64_t>(vocab_size_);
    }
    top_k_bytes =
        static_cast<std::size_t>(batch_size_) * sizeof(int64_t);
  } else {
    auto* top_k_int32 = reinterpret_cast<int32_t*>(slot->top_k);
    for (Tensor::Size row = 0; row < batch_size_; ++row) {
      const auto value = ReadTopK(top_k, row);
      top_k_int32[static_cast<std::size_t>(row)] =
          value > 0 && value <= static_cast<int64_t>(vocab_size_)
              ? static_cast<int32_t>(value)
              : static_cast<int32_t>(vocab_size_);
    }
    top_k_bytes =
        static_cast<std::size_t>(batch_size_) * sizeof(int32_t);
  }

  const void* staged_indices_data = indices ? indices->data() : nullptr;
  std::size_t indices_bytes{0};
  if (indices && indices->device().type() == Device::Type::kCpu) {
    if (indices->dtype() == DataType::kInt32) {
      for (Tensor::Size row = 0; row < batch_size_; ++row) {
        const auto value = static_cast<const int32_t*>(indices->data())[row];
        Require(value >= 0 && value < logits_batch_size_,
                "FlashInferSampling received an out-of-range host index");
      }
    } else {
      for (Tensor::Size row = 0; row < batch_size_; ++row) {
        const auto value = static_cast<const int64_t*>(indices->data())[row];
        Require(value >= 0 && value < logits_batch_size_,
                "FlashInferSampling received an out-of-range host index");
      }
    }
    indices_bytes = static_cast<std::size_t>(batch_size_) *
                    kDataTypeToSize.at(indices->dtype());
    std::memcpy(slot->indices, indices->data(), indices_bytes);
    staged_indices_data = indices_device;
  }

  StagingRecorder staging_recorder{static_cast<cudaEvent_t>(slot->event),
                                   stream, &slot->event_recorded};
  CheckCuda(
      cudaMemcpyAsync(top_p_device, slot->top_p,
                      static_cast<std::size_t>(batch_size_) * sizeof(float),
                      cudaMemcpyHostToDevice, stream),
      "FlashInferSampling failed to stage top-p values");
  CheckCuda(cudaMemcpyAsync(top_k_device, slot->top_k, top_k_bytes,
                            cudaMemcpyHostToDevice, stream),
            "FlashInferSampling failed to stage top-k values");
  if (indices_bytes != 0) {
    CheckCuda(cudaMemcpyAsync(indices_device, slot->indices, indices_bytes,
                              cudaMemcpyHostToDevice, stream),
              "FlashInferSampling failed to stage indices");
  }
  staging_recorder.Record();

  DispatchFunc<Device::Type::kNvidia, AllFloatTypes>(
      logits.dtype(),
      [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (!indices) {
          CastLogits<<<Blocks(matrix_elements), kThreads, 0, stream>>>(
              matrix_a, static_cast<const T*>(logits.data()), matrix_elements);
        } else if (indices->dtype() == DataType::kInt32) {
          GatherCastLogits<<<Blocks(matrix_elements), kThreads, 0, stream>>>(
              matrix_a, static_cast<const T*>(logits.data()),
              static_cast<const int32_t*>(staged_indices_data),
              static_cast<std::size_t>(batch_size_),
              static_cast<std::size_t>(logits_batch_size_),
              static_cast<std::size_t>(vocab_size_));
        } else {
          GatherCastLogits<<<Blocks(matrix_elements), kThreads, 0, stream>>>(
              matrix_a, static_cast<const T*>(logits.data()),
              static_cast<const int64_t*>(staged_indices_data),
              static_cast<std::size_t>(batch_size_),
              static_cast<std::size_t>(logits_batch_size_),
              static_cast<std::size_t>(vocab_size_));
        }
      },
      "`FlashInferSampling` logits cast");
  CheckCuda(cudaGetLastError(),
            "FlashInferSampling logits preparation kernel launch failed");

  int64_t matrix_shape[2]{static_cast<int64_t>(batch_size_),
                          static_cast<int64_t>(vocab_size_)};
  int64_t batch_shape[1]{static_cast<int64_t>(batch_size_)};
  int64_t scratch_shape[1]{static_cast<int64_t>(kScratchBytes)};
  auto matrix_a_tensor = MakeTensor(matrix_a, device_index_, 2, matrix_shape,
                                    Dtype(DataType::kFloat32));
  auto matrix_b_tensor = MakeTensor(matrix_b, device_index_, 2, matrix_shape,
                                    Dtype(DataType::kFloat32));
  auto top_k_tensor =
      MakeTensor(top_k_device, device_index_, 1, batch_shape,
                 Dtype(top_k_is_int64 ? DataType::kInt64 : DataType::kInt32));
  auto top_p_tensor = MakeTensor(top_p_device, device_index_, 1, batch_shape,
                                 Dtype(DataType::kFloat32));
  auto valid_tensor =
      MakeTensor(valid_device, device_index_, 1, batch_shape, {kDLBool, 8, 1});
  auto scratch_tensor = MakeTensor(scratch_device, device_index_, 1,
                                   scratch_shape, {kDLUInt, 8, 1});
  auto output_tensor =
      MakeTensor(out.data(), device_index_, 1, batch_shape, Dtype(out.dtype()));
  const auto actual_seed = static_cast<uint64_t>(
      seed.value_or(static_cast<int64_t>(std::random_device{}())));
  const auto actual_offset = static_cast<uint64_t>(offset.value_or(0));
  if (filter_apply_order == "top_k_first") {
    CheckCuda(cudaMemsetAsync(scratch_device, 0, kScratchBytes, stream),
              "FlashInferSampling failed to initialize row-state workspace");
    CallTopKMask(&matrix_a_tensor, &matrix_b_tensor, &top_k_tensor,
                 &scratch_tensor);
    CallSoftmax(&scratch_tensor, &matrix_b_tensor, &matrix_a_tensor);
    CallTopP(&matrix_a_tensor, &output_tensor, &valid_tensor, nullptr,
             &top_p_tensor, deterministic, actual_seed, actual_offset);
  } else {
    CallSoftmax(&scratch_tensor, &matrix_a_tensor, &matrix_b_tensor);
    CallJoint(&matrix_b_tensor, &output_tensor, &valid_tensor, nullptr,
              &top_k_tensor, &top_p_tensor, deterministic, actual_seed,
              actual_offset);
  }

  CheckCuda(cudaGetLastError(),
            "FlashInferSampling CUDA kernel launch failed");
}

}  // namespace infini::ops
