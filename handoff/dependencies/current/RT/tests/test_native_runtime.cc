#include <infini/rt.h>
#include INFINI_RT_TEST_RUNTIME_HEADER

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "test_helper.h"

namespace {

using Runtime = infini::rt::runtime::Runtime<INFINI_RT_TEST_DEVICE_TYPE>;

constexpr bool kExpectAsyncMemcpySuccess =
    INFINI_RT_TEST_EXPECT_ASYNC_MEMCPY_SUCCESS != 0;
constexpr bool kSupportsHostMemory = INFINI_RT_TEST_SUPPORTS_HOST_MEMORY != 0;
constexpr bool kSupportsAsyncMemory = INFINI_RT_TEST_SUPPORTS_ASYNC_MEMORY != 0;
constexpr bool kSupportsMemGetInfo = INFINI_RT_TEST_SUPPORTS_MEM_GET_INFO != 0;
constexpr bool kSupportsMemsetAsync = INFINI_RT_TEST_SUPPORTS_MEMSET_ASYNC != 0;
constexpr bool kSupportsStreamWaitEvent =
    INFINI_RT_TEST_SUPPORTS_STREAM_WAIT_EVENT != 0;
constexpr bool kSupportsEvent = INFINI_RT_TEST_SUPPORTS_EVENT != 0;
constexpr bool kSupportsEventElapsedTime =
    INFINI_RT_TEST_SUPPORTS_EVENT_ELAPSED_TIME != 0;

void ExpectSuccess(infini::rt::test::TestContext* context,
                   typename Runtime::Error status, const char* message) {
  context->Expect(status == Runtime::kSuccess, message);
}

void ExpectFailure(infini::rt::test::TestContext* context,
                   typename Runtime::Error status, const char* message) {
  context->Expect(status != Runtime::kSuccess, message);
}

void ExpectStatus(infini::rt::test::TestContext* context,
                  typename Runtime::Error status, bool expect_success,
                  const char* message) {
  if (expect_success) {
    ExpectSuccess(context, status, message);
  } else {
    ExpectFailure(context, status, message);
  }
}

bool SelectDevice() {
  int device_count = 0;
  if (Runtime::GetDeviceCount(&device_count) != Runtime::kSuccess ||
      device_count <= 0) {
    std::cout << INFINI_RT_TEST_BACKEND_NAME
              << " runtime skipped: no available device." << std::endl;
    return false;
  }

  if (Runtime::SetDevice(0) != Runtime::kSuccess) {
    std::cout << INFINI_RT_TEST_BACKEND_NAME
              << " runtime skipped: device 0 is not available." << std::endl;
    return false;
  }

  return true;
}

bool CreateStream(infini::rt::test::TestContext* context,
                  typename Runtime::Stream* stream) {
  const auto status = Runtime::StreamCreate(stream);
  ExpectSuccess(context, status,
                INFINI_RT_TEST_BACKEND_NAME " runtime should create a stream.");
  return status == Runtime::kSuccess;
}

void DestroyStream(infini::rt::test::TestContext* context,
                   typename Runtime::Stream stream) {
  ExpectSuccess(context, Runtime::StreamDestroy(stream),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should destroy a stream.");
}

void TestDevice(infini::rt::test::TestContext* context) {
  int current_device = -1;
  ExpectSuccess(context, Runtime::GetDevice(&current_device),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should get the current device.");
  context->ExpectEqual(current_device, 0,
                       INFINI_RT_TEST_BACKEND_NAME
                       " runtime should keep the current device.");

  int device_count = 0;
  ExpectSuccess(context, Runtime::GetDeviceCount(&device_count),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should get the device count.");
  context->Expect(device_count > 0, INFINI_RT_TEST_BACKEND_NAME
                  " runtime should report at least one device.");
}

void TestMallocAndFree(infini::rt::test::TestContext* context) {
  void* ptr = nullptr;
  ExpectSuccess(context, Runtime::Malloc(&ptr, 16),
                INFINI_RT_TEST_BACKEND_NAME " runtime should allocate memory.");
  context->Expect(ptr != nullptr, INFINI_RT_TEST_BACKEND_NAME
                  " runtime allocation should produce a pointer.");
  if (ptr != nullptr) {
    ExpectSuccess(context, Runtime::Free(ptr),
                  INFINI_RT_TEST_BACKEND_NAME " runtime should free memory.");
  }
}

void TestHostMemory(infini::rt::test::TestContext* context) {
  void* ptr = nullptr;
  const auto malloc_status = Runtime::MallocHost(&ptr, 16);
  ExpectStatus(context, malloc_status, kSupportsHostMemory,
               INFINI_RT_TEST_BACKEND_NAME
               " runtime should report expected host memory support.");

  if constexpr (kSupportsHostMemory) {
    context->Expect(ptr != nullptr, INFINI_RT_TEST_BACKEND_NAME
                    " runtime host allocation should produce a pointer.");
    if (ptr != nullptr) {
      ExpectSuccess(context, Runtime::FreeHost(ptr),
                    INFINI_RT_TEST_BACKEND_NAME
                    " runtime should free host memory.");
    }
  }
}

void TestAsyncMemory(infini::rt::test::TestContext* context) {
  typename Runtime::Stream stream{};
  if (!CreateStream(context, &stream)) {
    return;
  }

  void* ptr = nullptr;
  const auto malloc_status = Runtime::MallocAsync(&ptr, 16, stream);
  ExpectStatus(context, malloc_status, kSupportsAsyncMemory,
               INFINI_RT_TEST_BACKEND_NAME
               " runtime should report expected async allocation support.");

  if constexpr (kSupportsAsyncMemory) {
    context->Expect(ptr != nullptr, INFINI_RT_TEST_BACKEND_NAME
                    " runtime async allocation should produce a pointer.");
    if (ptr != nullptr) {
      ExpectSuccess(context, Runtime::FreeAsync(ptr, stream),
                    INFINI_RT_TEST_BACKEND_NAME
                    " runtime should free async memory.");
      ExpectSuccess(context, Runtime::StreamSynchronize(stream),
                    INFINI_RT_TEST_BACKEND_NAME
                    " runtime should synchronize async memory operations.");
    }
  } else if (ptr != nullptr) {
    ExpectSuccess(context, Runtime::Free(ptr),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should free unexpected async allocation.");
  }

  DestroyStream(context, stream);
}

void TestMemGetInfo(infini::rt::test::TestContext* context) {
  std::size_t free = 0;
  std::size_t total = 0;
  const auto status = Runtime::MemGetInfo(&free, &total);
  ExpectStatus(context, status, kSupportsMemGetInfo,
               INFINI_RT_TEST_BACKEND_NAME
               " runtime should report expected memory info support.");

  if constexpr (kSupportsMemGetInfo) {
    context->Expect(total > 0, INFINI_RT_TEST_BACKEND_NAME
                    " runtime should report total memory.");
    context->Expect(free <= total, INFINI_RT_TEST_BACKEND_NAME
                    " runtime free memory should not exceed total memory.");
  }
}

void TestMemcpyRoundTrip(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 8> input{0, 1, 2, 3, 4, 5, 6, 7};
  std::array<std::uint8_t, 8> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, Runtime::Malloc(&ptr, input.size()),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should allocate copy memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context,
                Runtime::Memcpy(ptr, input.data(), input.size(),
                                Runtime::kMemcpyHostToDevice),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should copy host data to runtime memory.");
  ExpectSuccess(context,
                Runtime::Memcpy(output.data(), ptr, output.size(),
                                Runtime::kMemcpyDeviceToHost),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should copy runtime memory to host.");
  ExpectSuccess(context, Runtime::Free(ptr),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should free copy memory.");

  context->ExpectEqual(output, input,
                       INFINI_RT_TEST_BACKEND_NAME
                       " runtime should preserve copied bytes.");
}

void TestMemcpyAsync(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 4> input{8, 9, 10, 11};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, Runtime::Malloc(&ptr, input.size()),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should allocate async copy memory.");
  if (ptr == nullptr) {
    return;
  }

  typename Runtime::Stream stream{};
  const auto async_status = Runtime::MemcpyAsync(
      ptr, input.data(), input.size(), Runtime::kMemcpyHostToDevice, stream);

  if constexpr (kExpectAsyncMemcpySuccess) {
    ExpectSuccess(context, async_status,
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should support async host-to-device copy.");
    ExpectSuccess(context, Runtime::DeviceSynchronize(),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should synchronize async host-to-device copy.");
    ExpectSuccess(context,
                  Runtime::Memcpy(output.data(), ptr, output.size(),
                                  Runtime::kMemcpyDeviceToHost),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should copy async data back to host.");
    context->ExpectEqual(output, input,
                         INFINI_RT_TEST_BACKEND_NAME
                         " runtime should preserve async copied bytes.");
  } else {
    ExpectFailure(context, async_status,
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should not report async memcpy success.");
  }

  ExpectSuccess(context, Runtime::Free(ptr),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should free async copy memory.");
}

void TestMemset(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, Runtime::Malloc(&ptr, output.size()),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should allocate memset memory.");
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context, Runtime::Memset(ptr, 0x5A, output.size()),
                INFINI_RT_TEST_BACKEND_NAME " runtime should fill memory.");
  ExpectSuccess(context,
                Runtime::Memcpy(output.data(), ptr, output.size(),
                                Runtime::kMemcpyDeviceToHost),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should copy filled memory to host.");
  ExpectSuccess(context, Runtime::Free(ptr),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should free memset memory.");

  for (const auto value : output) {
    context->ExpectEqual(value, static_cast<std::uint8_t>(0x5A),
                         INFINI_RT_TEST_BACKEND_NAME
                         " runtime should preserve filled bytes.");
  }
}

void TestMemsetAsync(infini::rt::test::TestContext* context) {
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, Runtime::Malloc(&ptr, output.size()),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should allocate async memset memory.");
  if (ptr == nullptr) {
    return;
  }

  typename Runtime::Stream stream{};
  if (!CreateStream(context, &stream)) {
    ExpectSuccess(context, Runtime::Free(ptr),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should free async memset memory.");
    return;
  }

  const auto memset_status =
      Runtime::MemsetAsync(ptr, 0xA5, output.size(), stream);
  ExpectStatus(context, memset_status, kSupportsMemsetAsync,
               INFINI_RT_TEST_BACKEND_NAME
               " runtime should report expected async memset support.");

  if constexpr (kSupportsMemsetAsync) {
    ExpectSuccess(context, Runtime::StreamSynchronize(stream),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should synchronize async memset.");
    ExpectSuccess(context,
                  Runtime::Memcpy(output.data(), ptr, output.size(),
                                  Runtime::kMemcpyDeviceToHost),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should copy async filled memory to host.");
    for (const auto value : output) {
      context->ExpectEqual(value, static_cast<std::uint8_t>(0xA5),
                           INFINI_RT_TEST_BACKEND_NAME
                           " runtime should preserve async filled bytes.");
    }
  }

  DestroyStream(context, stream);
  ExpectSuccess(context, Runtime::Free(ptr),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should free async memset memory.");
}

void TestStream(infini::rt::test::TestContext* context) {
  typename Runtime::Stream stream{};
  if (!CreateStream(context, &stream)) {
    return;
  }

  ExpectSuccess(context, Runtime::StreamSynchronize(stream),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should synchronize a stream.");
  DestroyStream(context, stream);
}

void TestEvent(infini::rt::test::TestContext* context) {
  typename Runtime::Event event{};
  const auto create_status = Runtime::EventCreate(&event);
  ExpectStatus(context, create_status, kSupportsEvent,
               INFINI_RT_TEST_BACKEND_NAME
               " runtime should report expected event support.");

  if constexpr (!kSupportsEvent) {
    return;
  }

  context->Expect(event != typename Runtime::Event{},
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime event creation should produce an event.");
  if (event == typename Runtime::Event{}) {
    return;
  }

  ExpectSuccess(context,
                Runtime::EventRecord(event, typename Runtime::Stream{}),
                INFINI_RT_TEST_BACKEND_NAME " runtime should record an event.");
  ExpectSuccess(context, Runtime::EventSynchronize(event),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should synchronize an event.");
  ExpectSuccess(context, Runtime::EventQuery(event),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should query a completed event.");
  ExpectSuccess(context, Runtime::EventDestroy(event),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should destroy an event.");

  typename Runtime::Event flagged_event{};
  ExpectSuccess(context, Runtime::EventCreateWithFlags(&flagged_event, 0),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should create an event with flags.");
  if (flagged_event != typename Runtime::Event{}) {
    ExpectSuccess(context, Runtime::EventDestroy(flagged_event),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should destroy a flagged event.");
  }
}

void TestStreamWaitEvent(infini::rt::test::TestContext* context) {
  typename Runtime::Stream stream{};
  if (!CreateStream(context, &stream)) {
    return;
  }

  typename Runtime::Event event{};
  if constexpr (kSupportsEvent) {
    ExpectSuccess(context, Runtime::EventCreate(&event),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should create a stream wait event.");
    if (event != typename Runtime::Event{}) {
      ExpectSuccess(context, Runtime::EventRecord(event, stream),
                    INFINI_RT_TEST_BACKEND_NAME
                    " runtime should record a stream wait event.");
    }
  }

  const auto wait_status = Runtime::StreamWaitEvent(stream, event, 0);
  ExpectStatus(context, wait_status, kSupportsStreamWaitEvent,
               INFINI_RT_TEST_BACKEND_NAME
               " runtime should report expected stream wait event support.");
  ExpectSuccess(context, Runtime::StreamSynchronize(stream),
                INFINI_RT_TEST_BACKEND_NAME
                " runtime should synchronize after stream wait event.");

  if constexpr (kSupportsEvent) {
    if (event != typename Runtime::Event{}) {
      ExpectSuccess(context, Runtime::EventDestroy(event),
                    INFINI_RT_TEST_BACKEND_NAME
                    " runtime should destroy a stream wait event.");
    }
  }
  DestroyStream(context, stream);
}

void TestEventElapsedTime(infini::rt::test::TestContext* context) {
  typename Runtime::Event start{};
  typename Runtime::Event end{};
  if constexpr (kSupportsEvent) {
    ExpectSuccess(context, Runtime::EventCreate(&start),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should create an elapsed-time start event.");
    ExpectSuccess(context, Runtime::EventCreate(&end),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should create an elapsed-time end event.");
  }

  if constexpr (kSupportsEventElapsedTime) {
    ExpectSuccess(context,
                  Runtime::EventRecord(start, typename Runtime::Stream{}),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should record an elapsed-time start event.");
    ExpectSuccess(context,
                  Runtime::EventRecord(end, typename Runtime::Stream{}),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should record an elapsed-time end event.");
    ExpectSuccess(context, Runtime::EventSynchronize(end),
                  INFINI_RT_TEST_BACKEND_NAME
                  " runtime should synchronize an elapsed-time event.");
  }

  float elapsed_ms = 0.0f;
  const auto elapsed_status =
      Runtime::EventElapsedTime(&elapsed_ms, start, end);
  ExpectStatus(context, elapsed_status, kSupportsEventElapsedTime,
               INFINI_RT_TEST_BACKEND_NAME
               " runtime should report expected event elapsed-time support.");

  if constexpr (kSupportsEvent) {
    if (start != typename Runtime::Event{}) {
      ExpectSuccess(context, Runtime::EventDestroy(start),
                    INFINI_RT_TEST_BACKEND_NAME
                    " runtime should destroy an elapsed-time start event.");
    }
    if (end != typename Runtime::Event{}) {
      ExpectSuccess(context, Runtime::EventDestroy(end),
                    INFINI_RT_TEST_BACKEND_NAME
                    " runtime should destroy an elapsed-time end event.");
    }
  }
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  if (!SelectDevice()) {
    return context.ExitCode();
  }

  TestDevice(&context);
  TestMallocAndFree(&context);
  TestHostMemory(&context);
  TestAsyncMemory(&context);
  TestMemGetInfo(&context);
  TestMemcpyRoundTrip(&context);
  TestMemcpyAsync(&context);
  TestMemset(&context);
  TestMemsetAsync(&context);
  TestStream(&context);
  TestEvent(&context);
  TestStreamWaitEvent(&context);
  TestEventElapsedTime(&context);

  return context.ExitCode();
}
