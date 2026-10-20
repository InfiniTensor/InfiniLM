#ifndef INFINI_RT_CPU_RUNTIME__H_
#define INFINI_RT_CPU_RUNTIME__H_

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "runtime.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kCpu> : RuntimeBase<Runtime<Device::Type::kCpu>> {
  static constexpr Device::Type kDeviceType = Device::Type::kCpu;

  using Error = int;

  using Stream = void*;

  using Event = void*;

  static constexpr Error kSuccess = 0;

  static constexpr Error kErrorInvalidValue = 1;

  static constexpr Error kErrorMemoryAllocation = 2;

  static Error SetDevice(int device) {
    if (device != 0) {
      return kErrorInvalidValue;
    }

    return kSuccess;
  }

  static Error GetDevice(int* device) {
    if (device == nullptr) {
      return kErrorInvalidValue;
    }

    *device = 0;

    return kSuccess;
  }

  static Error GetDeviceCount(int* count) {
    if (count == nullptr) {
      return kErrorInvalidValue;
    }

    *count = 1;

    return kSuccess;
  }

  static Error DeviceSynchronize() { return kSuccess; }

  static Error Malloc(void** ptr, std::size_t size) {
    if (ptr == nullptr) {
      return kErrorInvalidValue;
    }

    *ptr = std::malloc(size);

    if (size != 0 && *ptr == nullptr) {
      return kErrorMemoryAllocation;
    }

    return kSuccess;
  }

  static Error MallocHost(void** ptr, std::size_t size) {
    return Malloc(ptr, size);
  }

  static Error MallocAsync(void** ptr, std::size_t size, Stream) {
    return kErrorInvalidValue;
  }

  static Error Free(void* ptr) {
    std::free(ptr);

    return kSuccess;
  }

  static Error FreeHost(void* ptr) { return Free(ptr); }

  static Error FreeAsync(void* ptr, Stream) { return kErrorInvalidValue; }

  static Error MemGetInfo(std::size_t* free, std::size_t* total) {
    if (free == nullptr || total == nullptr) {
      return kErrorInvalidValue;
    }

    *free = 0;
    *total = 0;

#ifndef _WIN32
    FILE* fp = std::fopen("/proc/meminfo", "r");
    if (fp == nullptr) {
      return kErrorInvalidValue;
    }

    char label[64];
    std::size_t value = 0;
    while (std::fscanf(fp, "%63s %zu %*s", label, &value) == 2) {
      if (std::strcmp(label, "MemTotal:") == 0) {
        *total = value * 1024;
      } else if (std::strcmp(label, "MemAvailable:") == 0) {
        *free = value * 1024;
      }
    }
    std::fclose(fp);
#endif

    return *total == 0 ? kErrorInvalidValue : kSuccess;
  }

  static Error Memcpy(void* dst, const void* src, std::size_t size, int) {
    if ((dst == nullptr || src == nullptr) && size != 0) {
      return kErrorInvalidValue;
    }

    std::memcpy(dst, src, size);

    return kSuccess;
  }

  static Error MemcpyAsync(void* dst, const void* src, std::size_t size,
                           int kind, Stream) {
    return kErrorInvalidValue;
  }

  static Error Memset(void* ptr, int value, std::size_t count) {
    if (ptr == nullptr && count != 0) {
      return kErrorInvalidValue;
    }

    std::memset(ptr, value, count);

    return kSuccess;
  }

  static Error MemsetAsync(void* ptr, int value, std::size_t count, Stream) {
    return kErrorInvalidValue;
  }

  static Error StreamCreate(Stream* stream) {
    if (stream == nullptr) {
      return kErrorInvalidValue;
    }

    *stream = nullptr;

    return kSuccess;
  }

  static Error StreamDestroy(Stream) { return kSuccess; }

  static Error StreamSynchronize(Stream) { return kSuccess; }

  static Error StreamWaitEvent(Stream, Event, unsigned int) { return kSuccess; }

  using CpuEvent = std::chrono::steady_clock::time_point;

  static Error EventCreate(Event* event) {
    if (event == nullptr) {
      return kErrorInvalidValue;
    }

    *event = new (std::nothrow) CpuEvent(std::chrono::steady_clock::now());

    return *event == nullptr ? kErrorMemoryAllocation : kSuccess;
  }

  static Error EventCreateWithFlags(Event* event, unsigned int) {
    return EventCreate(event);
  }

  static Error EventRecord(Event event, Stream) {
    if (event == nullptr) {
      return kErrorInvalidValue;
    }

    *static_cast<CpuEvent*>(event) = std::chrono::steady_clock::now();

    return kSuccess;
  }

  static Error EventQuery(Event event) {
    return event == nullptr ? kErrorInvalidValue : kSuccess;
  }

  static Error EventSynchronize(Event event) {
    return event == nullptr ? kErrorInvalidValue : kSuccess;
  }

  static Error EventDestroy(Event event) {
    delete static_cast<CpuEvent*>(event);

    return kSuccess;
  }

  static Error EventElapsedTime(float* ms, Event start, Event end) {
    if (ms == nullptr || start == nullptr || end == nullptr) {
      return kErrorInvalidValue;
    }

    const auto* start_time = static_cast<const CpuEvent*>(start);
    const auto* end_time = static_cast<const CpuEvent*>(end);
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        *end_time - *start_time);
    *ms = static_cast<float>(duration.count()) / 1000.0f;

    return kSuccess;
  }

  static constexpr int kMemcpyHostToHost = 0;

  static constexpr int kMemcpyHostToDevice = 1;

  static constexpr int kMemcpyDeviceToHost = 2;

  static constexpr int kMemcpyDeviceToDevice = 3;
};

static_assert(Runtime<Device::Type::kCpu>::Validate());

}  // namespace infini::rt::runtime

#endif
