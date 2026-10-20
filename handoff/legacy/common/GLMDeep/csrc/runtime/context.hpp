#pragma once

#include "runtime/device.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::runtime {
class Runtime;

namespace context {
void setDevice(Device device);
Device getDevice();
size_t getDeviceCount(Device::Type type);
std::shared_ptr<Runtime> currentRuntime();
// Device-only frontend boundary; neither selects nor changes the recording owner.
std::shared_ptr<Runtime> resolveRuntime(Device device);

} // namespace context

} // namespace infinilm::runtime
