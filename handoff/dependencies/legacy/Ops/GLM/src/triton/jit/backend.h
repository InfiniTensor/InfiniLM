#ifndef INFINI_OPS_TRITON_JIT_BACKEND_H_
#define INFINI_OPS_TRITON_JIT_BACKEND_H_

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>

#include "device.h"
#include "runtime.h"

namespace infini::ops::triton::jit {

struct Grid {
  unsigned x{1};

  unsigned y{1};

  unsigned z{1};
};

namespace detail {

struct Target {
  std::string backend;

  std::string architecture;

  int warp_size{0};
};

enum class DataType {
  kPointer,
  kInt8,
  kUInt8,
  kInt16,
  kUInt16,
  kInt32,
  kUInt32,
  kInt64,
  kUInt64,
  kFloat32,
  kFloat64,
};

struct Argument {
  DataType type;

  std::uint64_t bits;
};

struct KernelMetadata {
  std::string name;

  unsigned num_warps{0};

  unsigned shared_memory_size{0};

  int global_scratch_size{0};

  int profile_scratch_size{0};
};

template <Device::Type kDev>
class Kernel {
 public:
  using Driver = ::infini::rt::driver::Driver<kDev>;
  using Function = typename Driver::Function;
  using Module = typename Driver::Module;

  Kernel(Module module, Function function, const KernelMetadata& metadata)
      : module_(module),
        function_(function),
        num_warps_(metadata.num_warps),
        shared_memory_size_(metadata.shared_memory_size) {}

  Kernel(const Kernel&) = delete;

  Kernel& operator=(const Kernel&) = delete;

  ~Kernel() {
    const auto status = Driver::ModuleUnload(module_);
    assert(status == Driver::kSuccess &&
           "Triton JIT failed to unload a kernel module.");
    (void)status;
  }

  Function function() const { return function_; }

  unsigned num_warps() const { return num_warps_; }

  unsigned shared_memory_size() const { return shared_memory_size_; }

 private:
  Module module_;

  Function function_;

  unsigned num_warps_;

  unsigned shared_memory_size_;
};

template <Device::Type kDev>
struct BackendTraits;

template <>
struct BackendTraits<Device::Type::kNvidia> {
  static constexpr const char* kName = "cuda";
};

template <Device::Type kDev,
          typename = std::void_t<decltype(BackendTraits<kDev>::kName)>>
class Backend {
 public:
  using Driver = ::infini::rt::driver::Driver<kDev>;
  using Runtime = ::infini::rt::runtime::Runtime<kDev>;
  using Function = typename Driver::Function;
  using Module = typename Driver::Module;
  using Stream = typename Driver::Stream;

  static Target CurrentTarget() {
    const int device_id = CurrentDevice();
    if (device_id < 0) return {};

    int major = 0;
    auto status = Runtime::DeviceGetAttribute(
        &major, Runtime::kDevAttrComputeCapabilityMajor, device_id);
    assert(status == Runtime::kSuccess &&
           "Triton JIT failed to get the device compute capability.");
    if (status != Runtime::kSuccess) return {};

    int minor = 0;
    status = Runtime::DeviceGetAttribute(
        &minor, Runtime::kDevAttrComputeCapabilityMinor, device_id);
    assert(status == Runtime::kSuccess &&
           "Triton JIT failed to get the device compute capability.");
    if (status != Runtime::kSuccess) return {};

    int warp_size = 0;
    status = Runtime::DeviceGetAttribute(&warp_size, Runtime::kDevAttrWarpSize,
                                         device_id);
    assert(status == Runtime::kSuccess &&
           "Triton JIT failed to get the warp size.");
    if (status != Runtime::kSuccess) return {};

    return {BackendTraits<kDev>::kName, std::to_string(major * 10 + minor),
            warp_size};
  }

  static int CurrentDevice() {
    int device_id = 0;
    const auto status = Runtime::GetDevice(&device_id);
    assert(status == Runtime::kSuccess &&
           "Triton JIT failed to get the current device.");
    return status == Runtime::kSuccess ? device_id : -1;
  }

  static std::unique_ptr<Kernel<kDev>> LoadKernel(
      const std::string& binary, const KernelMetadata& metadata) {
    Module module{};
    auto status = Driver::ModuleLoadData(&module, binary.data());
    assert(status == Driver::kSuccess &&
           "Triton JIT failed to load a kernel module.");
    if (status != Driver::kSuccess) return nullptr;

    Function function{};
    status =
        Driver::ModuleGetFunction(&function, module, metadata.name.c_str());
    assert(status == Driver::kSuccess &&
           "Triton JIT failed to get a kernel function.");
    if (status != Driver::kSuccess) {
      Driver::ModuleUnload(module);
      return nullptr;
    }

    constexpr unsigned kDefaultSharedMemoryLimit = 48 * 1024;
    if (metadata.shared_memory_size > kDefaultSharedMemoryLimit) {
      int maximum_shared_memory = 0;
      status = Runtime::DeviceGetAttribute(
          &maximum_shared_memory, Runtime::kDevAttrMaxSharedMemoryPerBlockOptin,
          CurrentDevice());
      assert(status == Runtime::kSuccess &&
             "Triton JIT failed to get the shared memory limit.");
      if (status != Runtime::kSuccess) {
        Driver::ModuleUnload(module);
        return nullptr;
      }

      int static_shared_memory = 0;
      status = Driver::FuncGetAttribute(&static_shared_memory,
                                        Driver::kFuncAttributeSharedSizeBytes,
                                        function);
      assert(status == Driver::kSuccess &&
             "Triton JIT failed to get the static shared memory size.");
      if (status != Driver::kSuccess) {
        Driver::ModuleUnload(module);
        return nullptr;
      }

      const bool shared_memory_fits =
          maximum_shared_memory >= static_shared_memory &&
          metadata.shared_memory_size <=
              static_cast<unsigned>(maximum_shared_memory -
                                    static_shared_memory);
      assert(shared_memory_fits &&
             "Triton JIT kernel requires too much shared memory.");
      if (!shared_memory_fits) {
        Driver::ModuleUnload(module);
        return nullptr;
      }

      status =
          Driver::FuncSetCacheConfig(function, Driver::kFuncCachePreferShared);
      assert(status == Driver::kSuccess &&
             "Triton JIT failed to configure the kernel cache.");
      if (status != Driver::kSuccess) {
        Driver::ModuleUnload(module);
        return nullptr;
      }

      status = Driver::FuncSetAttribute(
          function, Driver::kFuncAttributeMaxDynamicSharedSizeBytes,
          maximum_shared_memory - static_shared_memory);
      assert(status == Driver::kSuccess &&
             "Triton JIT failed to configure dynamic shared memory.");
      if (status != Driver::kSuccess) {
        Driver::ModuleUnload(module);
        return nullptr;
      }
    }

    return std::make_unique<Kernel<kDev>>(module, function, metadata);
  }

  static void Launch(const Kernel<kDev>& kernel, Grid grid, int warp_size,
                     void* stream, void** arguments) {
    const auto status = Driver::LaunchKernel(
        kernel.function(), grid.x, grid.y, grid.z,
        kernel.num_warps() * warp_size, 1, 1, kernel.shared_memory_size(),
        static_cast<Stream>(stream), arguments, nullptr);
    assert(status == Driver::kSuccess && "Triton JIT kernel launch failed.");
    (void)status;
  }
};

template <Device::Type kDev>
class ScopedDevice {
 public:
  using Runtime = ::infini::rt::runtime::Runtime<kDev>;

  static std::optional<ScopedDevice> Create(int device_id) {
    int previous_device_id = 0;
    auto status = Runtime::GetDevice(&previous_device_id);
    assert(status == Runtime::kSuccess &&
           "Triton JIT failed to get the current device.");
    if (status != Runtime::kSuccess) return std::nullopt;

    if (previous_device_id == device_id) {
      return ScopedDevice{previous_device_id, false};
    }

    status = Runtime::SetDevice(device_id);
    assert(status == Runtime::kSuccess &&
           "Triton JIT failed to select a device.");
    if (status != Runtime::kSuccess) return std::nullopt;

    return ScopedDevice{previous_device_id, true};
  }

  ScopedDevice(const ScopedDevice&) = delete;

  ScopedDevice& operator=(const ScopedDevice&) = delete;

  ScopedDevice(ScopedDevice&& other) noexcept
      : previous_device_id_(other.previous_device_id_),
        restore_(other.restore_) {
    other.restore_ = false;
  }

  ~ScopedDevice() {
    if (!restore_) return;

    const auto status = Runtime::SetDevice(previous_device_id_);
    assert(status == Runtime::kSuccess &&
           "Triton JIT failed to restore the previous device.");
    (void)status;
  }

 private:
  ScopedDevice(int previous_device_id, bool restore)
      : previous_device_id_(previous_device_id), restore_(restore) {}

  int previous_device_id_{0};

  bool restore_{false};
};

}  // namespace detail

}  // namespace infini::ops::triton::jit

#endif
