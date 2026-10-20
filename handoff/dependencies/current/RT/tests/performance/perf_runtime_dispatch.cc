#include <infini/rt.h>

#include <cstddef>
#include <iostream>

#include "perf_common.h"

namespace {

namespace perf = infini::rt::perf;
namespace runtime = infini::rt::runtime;
using DefaultRuntime = runtime::generated_detail::DefaultErrorRuntime;

bool PrepareRuntime() {
  int device_count = 0;
  if (runtime::GetDeviceCount(&device_count) != runtime::kSuccess ||
      device_count <= 0) {
    std::cerr << "perf_runtime_dispatch skipped: no available device."
              << std::endl;
    return false;
  }

  if (runtime::SetDevice(0) != runtime::kSuccess) {
    std::cerr << "perf_runtime_dispatch skipped: device 0 is not available."
              << std::endl;
    return false;
  }

  return true;
}

}  // namespace

int main() {
  if (!PrepareRuntime()) {
    return 0;
  }

  constexpr std::size_t kFastIterations = 1000000;
  constexpr std::size_t kRuntimeCallIterations = 200000;
  const auto device_type = infini::rt::runtime_device_type();

  perf::RunBenchmark("perf_runtime_dispatch.runtime_device_type", {},
                     kFastIterations, "ns", [] {
                       const auto current = infini::rt::runtime_device_type();
                       perf::DoNotOptimize(current);
                     });

  perf::RunBenchmark("perf_runtime_dispatch.set_runtime_device_type", {},
                     kFastIterations, "ns", [device_type] {
                       infini::rt::set_runtime_device_type(device_type);
                       perf::DoNotOptimize(device_type);
                     });

  perf::RunBenchmark("perf_runtime_dispatch.dispatch_GetDevice", {},
                     kRuntimeCallIterations, "ns", [] {
                       int device = -1;
                       const auto status = runtime::GetDevice(&device);
                       perf::DoNotOptimize(status);
                       perf::DoNotOptimize(device);
                     });

  perf::RunBenchmark("perf_runtime_dispatch.direct_GetDevice", {},
                     kRuntimeCallIterations, "ns", [] {
                       int device = -1;
                       const auto status = DefaultRuntime::GetDevice(&device);
                       perf::DoNotOptimize(status);
                       perf::DoNotOptimize(device);
                     });

  perf::RunBenchmark("perf_runtime_dispatch.dispatch_DeviceSynchronize", {},
                     kRuntimeCallIterations, "ns", [] {
                       const auto status = runtime::DeviceSynchronize();
                       perf::DoNotOptimize(status);
                     });

  perf::RunBenchmark("perf_runtime_dispatch.direct_DeviceSynchronize", {},
                     kRuntimeCallIterations, "ns", [] {
                       const auto status = DefaultRuntime::DeviceSynchronize();
                       perf::DoNotOptimize(status);
                     });

  return 0;
}
