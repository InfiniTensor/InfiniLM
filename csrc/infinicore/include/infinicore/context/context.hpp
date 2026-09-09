#pragma once

#include "../device.hpp"
#include "../memory.hpp"

#include "../graph/graph.hpp"

#include <infini/rt.h>

#include <memory>

namespace infinicore {

namespace context {
void setDevice(Device device);
Device getDevice();
size_t getDeviceCount(Device::Type type);

infini::rt::runtime::Stream getStream();

void syncStream();
void syncDevice();
void trimMemory();

std::shared_ptr<Memory> allocateMemory(size_t size);
std::shared_ptr<Memory> allocateHostMemory(size_t size);
std::shared_ptr<Memory> allocatePinnedHostMemory(size_t size);

void memcpyH2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyD2H(void *dst, const void *src, size_t size);
void memcpyD2D(void *dst, const void *src, size_t size, bool async = true);
void memcpyH2H(void *dst, const void *src, size_t size);

void setDeviceMemory(void *ptr, int value, size_t count);
void setDeviceMemoryAsync(void *ptr, int value, size_t count, infini::rt::runtime::Stream stream);

// Timing APIs for performance measurement
infini::rt::runtime::Event createEvent();
infini::rt::runtime::Event createEventWithFlags(uint32_t flags);
void recordEvent(infini::rt::runtime::Event event, infini::rt::runtime::Stream stream = nullptr);
bool queryEvent(infini::rt::runtime::Event event);
void synchronizeEvent(infini::rt::runtime::Event event);
void destroyEvent(infini::rt::runtime::Event event);
float elapsedTime(infini::rt::runtime::Event start, infini::rt::runtime::Event end);
void streamWaitEvent(infini::rt::runtime::Stream stream, infini::rt::runtime::Event event);

// Graph recording APIs
bool isGraphRecording();
void startGraphRecording();
void addGraphOperator(std::shared_ptr<graph::GraphOperator> op);
std::shared_ptr<graph::Graph> stopGraphRecording();
void cancelGraphRecording() noexcept;

} // namespace context

} // namespace infinicore
