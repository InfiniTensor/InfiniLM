#include <infini/rt.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;
namespace runtime = infini::rt::runtime;

bool Success(runtime::Error status) { return status == runtime::kSuccess; }

bool PrepareRuntime() {
  int device_count = 0;
  if (!Success(runtime::GetDeviceCount(&device_count)) || device_count <= 0) {
    std::cerr << "perf_memory skipped: no available device." << std::endl;
    return false;
  }

  if (!Success(runtime::SetDevice(0))) {
    std::cerr << "perf_memory skipped: device 0 is not available." << std::endl;
    return false;
  }

  return true;
}

std::vector<std::size_t> TestSizes() {
  std::vector<std::size_t> sizes{
      4 * 1024,
      64 * 1024,
      1024 * 1024,
      16 * 1024 * 1024,
  };

  if (std::getenv("INFINI_RT_PERF_ENABLE_LARGE") != nullptr) {
    sizes.push_back(256ULL * 1024ULL * 1024ULL);
  }

  return sizes;
}

std::size_t IterationsForSize(std::size_t size) {
  if (size <= 4 * 1024) {
    return 2000;
  }
  if (size <= 64 * 1024) {
    return 1000;
  }
  if (size <= 1024 * 1024) {
    return 200;
  }
  if (size <= 16 * 1024 * 1024) {
    return 20;
  }
  return 5;
}

std::vector<perf::Param> SizeParam(std::size_t size) {
  return {perf::NumberParam("size_bytes", static_cast<std::uint64_t>(size))};
}

bool CanMalloc(std::size_t size) {
  void* ptr = nullptr;
  const auto status = runtime::Malloc(&ptr, size);
  if (!Success(status)) {
    return false;
  }

  if (ptr != nullptr) {
    runtime::Free(ptr);
  }
  return true;
}

bool CanMallocHost(std::size_t size) {
  void* ptr = nullptr;
  const auto status = runtime::MallocHost(&ptr, size);
  if (!Success(status)) {
    return false;
  }

  if (ptr != nullptr) {
    runtime::FreeHost(ptr);
  }
  return true;
}

void FreeIfNeeded(void* ptr) {
  if (ptr != nullptr) {
    runtime::Free(ptr);
  }
}

runtime::Error MemcpyHostToDeviceAsync(void* dst, const void* src,
                                       std::size_t size,
                                       runtime::Stream stream) {
  return runtime::MemcpyAsync(dst, src, size, runtime::kMemcpyHostToDevice,
                              stream);
}

void MeasureAllocators(std::size_t size) {
  const auto iterations = IterationsForSize(size);
  const auto params = SizeParam(size);

  if (CanMalloc(size)) {
    perf::RunBenchmark("perf_memory.MallocFree", params, iterations, "us",
                       [size] {
                         void* ptr = nullptr;
                         auto status = runtime::Malloc(&ptr, size);
                         perf::DoNotOptimize(status);
                         if (Success(status)) {
                           status = runtime::Free(ptr);
                           perf::DoNotOptimize(status);
                         }
                       });
  } else {
    perf::SkipBenchmark("perf_memory.MallocFree", "Malloc failed during probe");
  }

  if (CanMallocHost(size)) {
    perf::RunBenchmark("perf_memory.MallocHostFreeHost", params, iterations,
                       "us", [size] {
                         void* ptr = nullptr;
                         auto status = runtime::MallocHost(&ptr, size);
                         perf::DoNotOptimize(status);
                         if (Success(status)) {
                           status = runtime::FreeHost(ptr);
                           perf::DoNotOptimize(status);
                         }
                       });
  } else {
    perf::SkipBenchmark("perf_memory.MallocHostFreeHost",
                        "host allocation is not supported");
  }
}

void MeasureCopies(std::size_t size) {
  const auto iterations = IterationsForSize(size);
  const auto params = SizeParam(size);
  std::vector<std::uint8_t> host_src(size, 0x5A);
  std::vector<std::uint8_t> host_dst(size, 0);
  void* device_src = nullptr;
  void* device_dst = nullptr;

  if (!Success(runtime::Malloc(&device_src, size)) ||
      !Success(runtime::Malloc(&device_dst, size))) {
    FreeIfNeeded(device_src);
    FreeIfNeeded(device_dst);
    perf::SkipBenchmark("perf_memory.copy", "device allocation failed");
    return;
  }

  if (!Success(runtime::Memcpy(device_src, host_src.data(), size,
                               runtime::kMemcpyHostToDevice)) ||
      !Success(runtime::Memcpy(device_dst, host_src.data(), size,
                               runtime::kMemcpyHostToDevice))) {
    FreeIfNeeded(device_src);
    FreeIfNeeded(device_dst);
    perf::SkipBenchmark("perf_memory.copy", "initial copy failed");
    return;
  }

  perf::RunBenchmark(
      "perf_memory.MemcpyHostToDevice", params, iterations, "us", [&] {
        const auto status = runtime::Memcpy(device_src, host_src.data(), size,
                                            runtime::kMemcpyHostToDevice);
        perf::DoNotOptimize(status);
      });

  perf::RunBenchmark(
      "perf_memory.MemcpyDeviceToHost", params, iterations, "us", [&] {
        const auto status = runtime::Memcpy(host_dst.data(), device_src, size,
                                            runtime::kMemcpyDeviceToHost);
        perf::DoNotOptimize(status);
      });

  perf::RunBenchmark(
      "perf_memory.MemcpyDeviceToDevice", params, iterations, "us", [&] {
        const auto status = runtime::Memcpy(device_dst, device_src, size,
                                            runtime::kMemcpyDeviceToDevice);
        perf::DoNotOptimize(status);
      });

  perf::RunBenchmark("perf_memory.Memset", params, iterations, "us", [&] {
    const auto status = runtime::Memset(device_dst, 0xA5, size);
    perf::DoNotOptimize(status);
  });

  runtime::Stream stream{};
  if (Success(runtime::StreamCreate(&stream))) {
    auto async_status =
        runtime::MemcpyAsync(device_src, host_src.data(), size,
                             runtime::kMemcpyHostToDevice, stream);
    if (Success(async_status)) {
      runtime::StreamSynchronize(stream);
      auto copy_async = [&] {
        auto status =
            MemcpyHostToDeviceAsync(device_src, host_src.data(), size, stream);
        if (Success(status)) {
          status = runtime::StreamSynchronize(stream);
        }
        perf::DoNotOptimize(status);
      };
      perf::RunBenchmark("perf_memory.MemcpyAsyncHostToDevice", params,
                         iterations, "us", copy_async);
    } else {
      perf::SkipBenchmark("perf_memory.MemcpyAsyncHostToDevice",
                          "async memcpy is not supported");
    }

    async_status = runtime::MemsetAsync(device_dst, 0xA5, size, stream);
    if (Success(async_status)) {
      runtime::StreamSynchronize(stream);
      perf::RunBenchmark(
          "perf_memory.MemsetAsync", params, iterations, "us", [&] {
            auto status = runtime::MemsetAsync(device_dst, 0xA5, size, stream);
            if (Success(status)) {
              status = runtime::StreamSynchronize(stream);
            }
            perf::DoNotOptimize(status);
          });
    } else {
      perf::SkipBenchmark("perf_memory.MemsetAsync",
                          "async memset is not supported");
    }

    void* async_ptr = nullptr;
    async_status = runtime::MallocAsync(&async_ptr, size, stream);
    if (Success(async_status)) {
      async_status = runtime::FreeAsync(async_ptr, stream);
      if (Success(async_status)) {
        runtime::StreamSynchronize(stream);
        perf::RunBenchmark("perf_memory.MallocAsyncFreeAsync", params,
                           iterations, "us", [size, stream] {
                             void* ptr = nullptr;
                             auto status =
                                 runtime::MallocAsync(&ptr, size, stream);
                             if (Success(status)) {
                               status = runtime::FreeAsync(ptr, stream);
                             }
                             if (Success(status)) {
                               status = runtime::StreamSynchronize(stream);
                             }
                             perf::DoNotOptimize(status);
                           });
      } else if (async_ptr != nullptr) {
        runtime::Free(async_ptr);
      }
    } else {
      perf::SkipBenchmark("perf_memory.MallocAsyncFreeAsync",
                          "async allocation is not supported");
    }

    runtime::StreamDestroy(stream);
  }

  FreeIfNeeded(device_src);
  FreeIfNeeded(device_dst);
}

}  // namespace

int main() {
  if (!PrepareRuntime()) {
    return 0;
  }

  for (const auto size : TestSizes()) {
    MeasureAllocators(size);
    MeasureCopies(size);
  }

  return 0;
}
