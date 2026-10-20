#include <infini/rt/device_guard.h>
#include <infini/rt/reusable_memory_pool.h>

#include <array>
#include <iostream>
#include <stdexcept>

#include "test_helper.h"

namespace {
using namespace infini::rt;

void Check(runtime::Error status) {
  if (status != runtime::kSuccess)
    throw std::runtime_error("runtime error " + std::to_string(status));
}

void TestCapture(test::TestContext& context) {
  DeviceGuard guard(Device::Type::kNvidia, 0, true);
  ReusableMemoryPool pool(Device{Device::Type::kNvidia, 0});
  auto* cached = pool.allocate(32);
  auto* exact = pool.allocate_exact(32);
  pool.deallocate(cached);
  pool.deallocate(exact);
  Check(runtime::StreamSynchronize(pool.stream()));

  pool.begin_pin_mode();
  Check(runtime::StreamBeginCapture(
      pool.stream(), runtime::StreamCaptureMode::kStreamCaptureModeRelaxed));
  auto* reused = pool.allocate(32);
  auto* reused_exact = pool.allocate_exact(32);
  context.Expect(reused == cached && reused_exact == exact,
                 "Capture must still reuse cached blocks.");
  for (bool exact_request : {false, true}) {
    bool rejected = false;
    try {
      auto* unexpected =
          exact_request ? pool.allocate_exact(1024) : pool.allocate(64 * 1024);
      pool.deallocate(unexpected);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    context.Expect(rejected, "Capture must reject a pool cache miss.");
  }
  // A rejected allocation must leave capture usable, not invalidated.
  Check(runtime::MemsetAsync(reused, 42, 32, pool.stream()));
  Check(runtime::MemsetAsync(reused_exact, 17, 32, pool.stream()));
  pool.deallocate(reused);
  pool.deallocate(reused_exact);
  runtime::Graph graph = nullptr;
  Check(runtime::StreamEndCapture(pool.stream(), &graph));
  auto lease = pool.commit_pin_mode();
  runtime::GraphExec exec = nullptr;
  Check(runtime::GraphInstantiate(&exec, graph));
  for (int replay = 0; replay < 3; ++replay) {
    Check(runtime::MemsetAsync(cached, 0, 32, pool.stream()));
    Check(runtime::MemsetAsync(exact, 0, 32, pool.stream()));
    Check(runtime::GraphLaunch(exec, pool.stream()));
    Check(runtime::StreamSynchronize(pool.stream()));
    std::array<unsigned char, 32> result{};
    Check(runtime::Memcpy(result.data(), cached, result.size(),
                          runtime::kMemcpyDeviceToHost));
    for (auto value : result)
      context.Expect(value == 42, "Replayed size-class buffer must be valid.");
    Check(runtime::Memcpy(result.data(), exact, result.size(),
                          runtime::kMemcpyDeviceToHost));
    for (auto value : result)
      context.Expect(value == 17, "Replayed exact buffer must be valid.");
  }
  Check(runtime::GraphExecDestroy(exec));
  Check(runtime::GraphDestroy(graph));
  lease.reset();
  pool.trim();

  // Pin mode alone is LM recording, not native CUDA stream capture.
  pool.begin_pin_mode();
  auto* outside_capture = pool.allocate(64 * 1024);
  pool.deallocate(outside_capture);
  pool.cancel_pin_mode();
  pool.trim();
}
}  // namespace

int main() {
  test::TestContext context;
  int count = 0;
  if (runtime::GetDeviceCount(Device::Type::kNvidia, &count) !=
          runtime::kSuccess ||
      count == 0) {
    std::cout << "Skipping pool capture test: no NVIDIA device.\n";
    return 0;
  }
  try {
    TestCapture(context);
  } catch (const std::exception& error) {
    context.Expect(false, error.what());
  }
  return context.ExitCode();
}
