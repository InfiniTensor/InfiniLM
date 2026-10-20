#ifndef INFINI_RT_DEVICE_GUARD_H_
#define INFINI_RT_DEVICE_GUARD_H_

#include <infini/rt.h>
#include <cstdio>
#include <stdexcept>

namespace infini::rt {

// Select the owning resource's device and restore this thread's prior selection.
class DeviceGuard {
 public:
  DeviceGuard(Device::Type target_type, int target_index) noexcept
      : previous_type_(runtime_device_type()) {
    const auto previous_status = runtime::GetDevice(&previous_index_);
    restore_index_ = previous_status == runtime::kSuccess;
    warn(previous_status, "read previous device");
    if (target_type == Device::Type::kCount) {
      std::fputs("InfiniRT device scope invalid target device type\n", stderr);
      restore();
      return;
    }
    if (restore_index_ && previous_type_ == target_type && previous_index_ == target_index) {
      active_ = true;
      restored_ = true;
      return;
    }
    set_runtime_device_type(target_type);
    const auto status = runtime::SetDevice(target_index);
    active_ = status == runtime::kSuccess;
    warn(status, "select target device");
    if (!active_) restore();
  }

  DeviceGuard(Device::Type target_type, int target_index, bool required)
      : DeviceGuard(target_type, target_index) {
    if (required && !active_) throw std::runtime_error("cannot select runtime device");
  }

  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard &operator=(const DeviceGuard &) = delete;
  DeviceGuard(DeviceGuard &&) = delete;
  DeviceGuard &operator=(DeviceGuard &&) = delete;
  ~DeviceGuard() noexcept { restore(); }

  bool active() const noexcept { return active_; }

 private:
  static void warn(runtime::Error status, const char *operation) noexcept {
    if (status != runtime::kSuccess)
      std::fprintf(stderr, "InfiniRT device scope %s error=%lld\n", operation,
                   static_cast<long long>(status));
  }
  void restore() noexcept {
    if (restored_) return;
    restored_ = true;
    set_runtime_device_type(previous_type_);
    if (restore_index_) warn(runtime::SetDevice(previous_index_), "restore previous device");
  }

  Device::Type previous_type_;
  int previous_index_ = 0;
  bool restore_index_ = false;
  bool active_ = false;
  bool restored_ = false;
};

}  // namespace infini::rt

#endif
