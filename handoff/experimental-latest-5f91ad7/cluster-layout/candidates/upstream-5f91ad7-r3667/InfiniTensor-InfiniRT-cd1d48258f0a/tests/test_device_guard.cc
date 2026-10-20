#include <infini/rt/device_guard.h>
#include <infini/rt/reusable_memory_pool.h>

#include <stdexcept>

#include "test_helper.h"

namespace {
using namespace infini::rt;
struct TestUnwind {};

void CheckRejected(test::TestContext& context, Device::Type type) {
  const auto previous = runtime_device_type();
  {
    DeviceGuard guard(type, 0);
    context.Expect(!guard.active(), "Unavailable device must not be active.");
    context.Expect(runtime_device_type() == previous,
                   "Failed selection must preserve the current backend.");
  }
  bool threw = false;
  try {
    DeviceGuard guard(type, 0, true);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  context.Expect(threw, "Required unavailable device must throw.");
  threw = false;
  try {
    ReusableMemoryPool pool(Device{type, 0});
  } catch (const std::exception&) {
    threw = true;
  }
  context.Expect(threw, "Pool must reject an unavailable device.");
  context.Expect(runtime_device_type() == previous,
                 "Rejected pool construction must preserve the backend.");
}

template <Device::Type type>
void CheckBackend(test::TestContext& context) {
  if constexpr (!DeviceEnabled<type>::value) {
    CheckRejected(context, type);
  } else {
    int count = 0;
    if (runtime::GetDeviceCount(type, &count) != runtime::kSuccess ||
        count == 0)
      return;
    // Exercise both the no-switch path and an index switch, where available.
    DeviceGuard outer(type, 0, true);
    {
      DeviceGuard inner(type, count - 1, true);
      int selected = -1;
      context.Expect(runtime::GetDevice(&selected) == runtime::kSuccess &&
                         selected == count - 1,
                     "Guard must select the requested device.");
    }
    int restored = -1;
    context.Expect(runtime_device_type() == type &&
                       runtime::GetDevice(&restored) == runtime::kSuccess &&
                       restored == 0,
                   "Nested guard must restore the prior selection.");

    for (bool unwind : {false, true}) {
      if (count > 1) {
        set_runtime_device_type(type);
        context.Expect(runtime::SetDevice(0) == runtime::kSuccess,
                       "Failed to select the initial device.");
        try {
          DeviceGuard same_device(type, 0, true);
          context.Expect(runtime::SetDevice(count - 1) == runtime::kSuccess,
                         "Failed to change device inside the guard.");
          if (unwind) throw TestUnwind{};
        } catch (const TestUnwind&) {
        }
        context.Expect(
            runtime::GetDevice(&restored) == runtime::kSuccess && restored == 0,
            "Same-device guard must restore an internal index change.");
      }
      if constexpr (type != Device::Type::kCpu &&
                    DeviceEnabled<Device::Type::kCpu>::value) {
        set_runtime_device_type(type);
        context.Expect(runtime::SetDevice(0) == runtime::kSuccess,
                       "Failed to select the initial backend.");
        try {
          DeviceGuard same_device(type, 0, true);
          set_runtime_device_type(Device::Type::kCpu);
          if (unwind) throw TestUnwind{};
        } catch (const TestUnwind&) {
        }
        context.Expect(
            runtime_device_type() == type &&
                runtime::GetDevice(&restored) == runtime::kSuccess &&
                restored == 0,
            "Same-device guard must restore an internal backend change.");
      }
    }
  }
}

template <Device::Type... types>
void CheckBackends(test::TestContext& context, List<types...>) {
  (CheckBackend<types>(context), ...);
}
}  // namespace

int main() {
  test::TestContext context;
  CheckRejected(context, Device::Type::kCount);
  CheckRejected(context, static_cast<Device::Type>(-1));
  CheckRejected(context, static_cast<Device::Type>(999));
  CheckBackends(context, AllDeviceTypes{});
  return context.ExitCode();
}
