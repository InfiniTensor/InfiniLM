#pragma once

#include "../device.hpp"
#include "../memory.hpp"

#include "../graph/graph.hpp"

#include <infinirt.h>

#include <memory>

namespace infinicore {

namespace context {
void setDevice(Device device);
Device getDevice();
size_t getDeviceCount(Device::Type type);

infinirtStream_t getStream();

void syncStream();
void syncDevice();

std::shared_ptr<Memory> allocateMemory(size_t size);
std::shared_ptr<Memory> allocateHostMemory(size_t size);
std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);

void memcpyH2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyD2H(void *dst, const void *src, size_t size);
void memcpyD2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyH2H(void *dst, const void *src, size_t size);

// Timing APIs for performance measurement
infinirtEvent_t createEvent();
infinirtEvent_t createEventWithFlags(uint32_t flags);
void recordEvent(infinirtEvent_t event, infinirtStream_t stream = nullptr);
bool queryEvent(infinirtEvent_t event);
void synchronizeEvent(infinirtEvent_t event);
void destroyEvent(infinirtEvent_t event);
float elapsedTime(infinirtEvent_t start, infinirtEvent_t end);
void streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

// Graph recording APIs
bool isGraphRecording();
void startGraphRecording();
void addGraphOperator(std::shared_ptr<graph::GraphOperator> op);
std::shared_ptr<graph::Graph> stopGraphRecording(const graph::GraphInstantiateFence &fence = nullptr);

} // namespace context

} // namespace infinicore
