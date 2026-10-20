#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

#include "test_helper.h"

namespace {

namespace runtime = infini::rt::runtime;

struct RuntimeApiExpectations {
  bool async_memcpy;
  bool host_memory;
  bool async_memory;
  bool mem_get_info;
  bool memset_async;
  bool stream_wait_event;
  bool event;
  bool event_elapsed_time;
};

void ExpectSuccess(infini::rt::test::TestContext* context,
                   runtime::Error status, std::string_view message) {
  context->Expect(status == runtime::kSuccess, message);
}

void ExpectFailure(infini::rt::test::TestContext* context,
                   runtime::Error status, std::string_view message) {
  context->Expect(status != runtime::kSuccess, message);
}

void ExpectStatus(infini::rt::test::TestContext* context, runtime::Error status,
                  bool expect_success, std::string_view message) {
  if (expect_success) {
    ExpectSuccess(context, status, message);
  } else {
    ExpectFailure(context, status, message);
  }
}

std::string Message(std::string_view backend_name, std::string_view message) {
  return std::string{backend_name} + " dispatch should " +
         std::string{message} + ".";
}

bool SelectDevice(infini::rt::test::TestContext* context,
                  infini::rt::Device::Type device_type,
                  const char* backend_name) {
  infini::rt::set_runtime_device_type(device_type);
  if (!context->Expect(infini::rt::runtime_device_type() == device_type,
                       Message(backend_name, "report the selected backend"))) {
    return false;
  }

  int device_count = 0;
  if (runtime::GetDeviceCount(&device_count) != runtime::kSuccess ||
      device_count <= 0) {
    std::cout << backend_name << " dispatch skipped: no available device."
              << std::endl;
    return false;
  }

  if (runtime::SetDevice(0) != runtime::kSuccess) {
    std::cout << backend_name << " dispatch skipped: device 0 is not available."
              << std::endl;
    return false;
  }

  return true;
}

bool CreateStream(infini::rt::test::TestContext* context,
                  const char* backend_name, runtime::Stream* stream) {
  const auto status = runtime::StreamCreate(stream);
  ExpectSuccess(context, status, Message(backend_name, "create a stream"));
  return status == runtime::kSuccess;
}

void DestroyStream(infini::rt::test::TestContext* context,
                   const char* backend_name, runtime::Stream stream) {
  ExpectSuccess(context, runtime::StreamDestroy(stream),
                Message(backend_name, "destroy a stream"));
}

void TestCoreDispatch(infini::rt::test::TestContext* context,
                      const char* backend_name,
                      const RuntimeApiExpectations& expectations) {
  std::array<std::uint8_t, 4> input{1, 2, 3, 4};
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  int current_device = -1;
  ExpectSuccess(context, runtime::GetDevice(&current_device),
                Message(backend_name, "get the current device"));
  context->ExpectEqual(current_device, 0,
                       Message(backend_name, "keep the current device"));

  ExpectSuccess(context, runtime::Malloc(&ptr, input.size()),
                Message(backend_name, "allocate memory"));
  if (ptr == nullptr) {
    return;
  }

  ExpectSuccess(context,
                runtime::Memcpy(ptr, input.data(), input.size(),
                                runtime::kMemcpyHostToDevice),
                Message(backend_name, "copy host data to runtime memory"));

  runtime::Stream stream{};
  const auto async_status = runtime::MemcpyAsync(
      ptr, input.data(), input.size(), runtime::kMemcpyHostToDevice, stream);
  ExpectStatus(context, async_status, expectations.async_memcpy,
               Message(backend_name, "report expected async memcpy support"));
  if (expectations.async_memcpy) {
    ExpectSuccess(context, runtime::DeviceSynchronize(),
                  Message(backend_name, "synchronize async copy"));
  }

  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::kMemcpyDeviceToHost),
                Message(backend_name, "copy runtime memory to host"));

  context->ExpectEqual(output, input,
                       Message(backend_name, "preserve copied bytes"));

  ExpectSuccess(context, runtime::Memset(ptr, 0x5A, output.size()),
                Message(backend_name, "fill runtime memory"));
  ExpectSuccess(context, runtime::DeviceSynchronize(),
                Message(backend_name, "synchronize filled memory"));
  ExpectSuccess(context,
                runtime::Memcpy(output.data(), ptr, output.size(),
                                runtime::kMemcpyDeviceToHost),
                Message(backend_name, "copy filled memory to host"));
  for (const auto value : output) {
    context->ExpectEqual(value, static_cast<std::uint8_t>(0x5A),
                         Message(backend_name, "preserve filled bytes"));
  }

  ExpectSuccess(context, runtime::Free(ptr),
                Message(backend_name, "free memory"));
}

void TestHostMemory(infini::rt::test::TestContext* context,
                    const char* backend_name,
                    const RuntimeApiExpectations& expectations) {
  void* ptr = nullptr;
  const auto malloc_status = runtime::MallocHost(&ptr, 16);
  ExpectStatus(context, malloc_status, expectations.host_memory,
               Message(backend_name, "report expected host memory support"));

  if (expectations.host_memory) {
    context->Expect(
        ptr != nullptr,
        Message(backend_name, "produce a pointer from host allocation"));
    if (ptr != nullptr) {
      ExpectSuccess(context, runtime::FreeHost(ptr),
                    Message(backend_name, "free host memory"));
    }
  }
}

void TestAsyncMemory(infini::rt::test::TestContext* context,
                     const char* backend_name,
                     const RuntimeApiExpectations& expectations) {
  runtime::Stream stream{};
  if (!CreateStream(context, backend_name, &stream)) {
    return;
  }

  void* ptr = nullptr;
  const auto malloc_status = runtime::MallocAsync(&ptr, 16, stream);
  ExpectStatus(
      context, malloc_status, expectations.async_memory,
      Message(backend_name, "report expected async allocation support"));

  if (expectations.async_memory) {
    context->Expect(
        ptr != nullptr,
        Message(backend_name, "produce a pointer from async allocation"));
    if (ptr != nullptr) {
      ExpectSuccess(context, runtime::FreeAsync(ptr, stream),
                    Message(backend_name, "free async memory"));
      ExpectSuccess(
          context, runtime::StreamSynchronize(stream),
          Message(backend_name, "synchronize async memory operations"));
    }
  } else if (ptr != nullptr) {
    ExpectSuccess(context, runtime::Free(ptr),
                  Message(backend_name, "free unexpected async allocation"));
  }

  DestroyStream(context, backend_name, stream);
}

void TestMemGetInfo(infini::rt::test::TestContext* context,
                    const char* backend_name,
                    const RuntimeApiExpectations& expectations) {
  std::size_t free = 0;
  std::size_t total = 0;
  const auto status = runtime::MemGetInfo(&free, &total);
  ExpectStatus(context, status, expectations.mem_get_info,
               Message(backend_name, "report expected memory info support"));

  if (expectations.mem_get_info) {
    context->Expect(total > 0, Message(backend_name, "report total memory"));
    context->Expect(
        free <= total,
        Message(backend_name, "report free memory not exceeding total memory"));
  }
}

void TestMemsetAsync(infini::rt::test::TestContext* context,
                     const char* backend_name,
                     const RuntimeApiExpectations& expectations) {
  std::array<std::uint8_t, 4> output{};
  void* ptr = nullptr;

  ExpectSuccess(context, runtime::Malloc(&ptr, output.size()),
                Message(backend_name, "allocate async memset memory"));
  if (ptr == nullptr) {
    return;
  }

  runtime::Stream stream{};
  if (!CreateStream(context, backend_name, &stream)) {
    ExpectSuccess(context, runtime::Free(ptr),
                  Message(backend_name, "free async memset memory"));
    return;
  }

  const auto memset_status =
      runtime::MemsetAsync(ptr, 0xA5, output.size(), stream);
  ExpectStatus(context, memset_status, expectations.memset_async,
               Message(backend_name, "report expected async memset support"));

  if (expectations.memset_async) {
    ExpectSuccess(context, runtime::StreamSynchronize(stream),
                  Message(backend_name, "synchronize async memset"));
    ExpectSuccess(context,
                  runtime::Memcpy(output.data(), ptr, output.size(),
                                  runtime::kMemcpyDeviceToHost),
                  Message(backend_name, "copy async filled memory to host"));
    for (const auto value : output) {
      context->ExpectEqual(
          value, static_cast<std::uint8_t>(0xA5),
          Message(backend_name, "preserve async filled bytes"));
    }
  }

  DestroyStream(context, backend_name, stream);
  ExpectSuccess(context, runtime::Free(ptr),
                Message(backend_name, "free async memset memory"));
}

void TestStream(infini::rt::test::TestContext* context,
                const char* backend_name) {
  runtime::Stream stream{};
  if (!CreateStream(context, backend_name, &stream)) {
    return;
  }

  ExpectSuccess(context, runtime::StreamSynchronize(stream),
                Message(backend_name, "synchronize a stream"));
  DestroyStream(context, backend_name, stream);
}

void TestEvent(infini::rt::test::TestContext* context, const char* backend_name,
               const RuntimeApiExpectations& expectations) {
  runtime::Event event{};
  const auto create_status = runtime::EventCreate(&event);
  ExpectStatus(context, create_status, expectations.event,
               Message(backend_name, "report expected event support"));

  if (!expectations.event) {
    return;
  }

  context->Expect(
      event != nullptr,
      Message(backend_name, "produce an event from event creation"));
  if (event == nullptr) {
    return;
  }

  ExpectSuccess(context, runtime::EventRecord(event, runtime::Stream{}),
                Message(backend_name, "record an event"));
  ExpectSuccess(context, runtime::EventSynchronize(event),
                Message(backend_name, "synchronize an event"));
  ExpectSuccess(context, runtime::EventQuery(event),
                Message(backend_name, "query a completed event"));
  ExpectSuccess(context, runtime::EventDestroy(event),
                Message(backend_name, "destroy an event"));

  runtime::Event flagged_event{};
  ExpectSuccess(context, runtime::EventCreateWithFlags(&flagged_event, 0),
                Message(backend_name, "create an event with flags"));
  if (flagged_event != nullptr) {
    ExpectSuccess(context, runtime::EventDestroy(flagged_event),
                  Message(backend_name, "destroy a flagged event"));
  }
}

void TestStreamWaitEvent(infini::rt::test::TestContext* context,
                         const char* backend_name,
                         const RuntimeApiExpectations& expectations) {
  runtime::Stream stream{};
  if (!CreateStream(context, backend_name, &stream)) {
    return;
  }

  runtime::Event event{};
  if (expectations.event) {
    ExpectSuccess(context, runtime::EventCreate(&event),
                  Message(backend_name, "create a stream wait event"));
    if (event != nullptr) {
      ExpectSuccess(context, runtime::EventRecord(event, stream),
                    Message(backend_name, "record a stream wait event"));
    }
  }

  const auto wait_status = runtime::StreamWaitEvent(stream, event, 0);
  ExpectStatus(
      context, wait_status, expectations.stream_wait_event,
      Message(backend_name, "report expected stream wait event support"));
  ExpectSuccess(context, runtime::StreamSynchronize(stream),
                Message(backend_name, "synchronize after stream wait event"));

  if (event != nullptr) {
    ExpectSuccess(context, runtime::EventDestroy(event),
                  Message(backend_name, "destroy a stream wait event"));
  }
  DestroyStream(context, backend_name, stream);
}

void TestEventElapsedTime(infini::rt::test::TestContext* context,
                          const char* backend_name,
                          const RuntimeApiExpectations& expectations) {
  runtime::Event start{};
  runtime::Event end{};
  if (expectations.event) {
    ExpectSuccess(context, runtime::EventCreate(&start),
                  Message(backend_name, "create an elapsed-time start event"));
    ExpectSuccess(context, runtime::EventCreate(&end),
                  Message(backend_name, "create an elapsed-time end event"));
  }

  if (expectations.event_elapsed_time) {
    ExpectSuccess(context, runtime::EventRecord(start, runtime::Stream{}),
                  Message(backend_name, "record an elapsed-time start event"));
    ExpectSuccess(context, runtime::EventRecord(end, runtime::Stream{}),
                  Message(backend_name, "record an elapsed-time end event"));
    ExpectSuccess(context, runtime::EventSynchronize(end),
                  Message(backend_name, "synchronize an elapsed-time event"));
  }

  float elapsed_ms = 0.0f;
  const auto elapsed_status =
      runtime::EventElapsedTime(&elapsed_ms, start, end);
  ExpectStatus(
      context, elapsed_status, expectations.event_elapsed_time,
      Message(backend_name, "report expected event elapsed-time support"));

  if (start != nullptr) {
    ExpectSuccess(context, runtime::EventDestroy(start),
                  Message(backend_name, "destroy an elapsed-time start event"));
  }
  if (end != nullptr) {
    ExpectSuccess(context, runtime::EventDestroy(end),
                  Message(backend_name, "destroy an elapsed-time end event"));
  }
}

void TestDispatch(infini::rt::test::TestContext* context,
                  infini::rt::Device::Type device_type,
                  const char* backend_name,
                  const RuntimeApiExpectations& expectations) {
  if (!SelectDevice(context, device_type, backend_name)) {
    return;
  }

  TestCoreDispatch(context, backend_name, expectations);
  TestHostMemory(context, backend_name, expectations);
  TestAsyncMemory(context, backend_name, expectations);
  TestMemGetInfo(context, backend_name, expectations);
  TestMemsetAsync(context, backend_name, expectations);
  TestStream(context, backend_name);
  TestEvent(context, backend_name, expectations);
  TestStreamWaitEvent(context, backend_name, expectations);
  TestEventElapsedTime(context, backend_name, expectations);
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

#if defined(INFINI_RT_TEST_WITH_CPU)
  TestDispatch(&context, infini::rt::Device::Type::kCpu, "CPU",
               {false, true, false, true, false, true, true, true});
#endif

#if defined(INFINI_RT_TEST_WITH_NVIDIA)
  TestDispatch(&context, infini::rt::Device::Type::kNvidia, "NVIDIA",
               {true, true, true, true, true, true, true, true});
#endif

#if defined(INFINI_RT_TEST_WITH_ILUVATAR)
  TestDispatch(&context, infini::rt::Device::Type::kIluvatar, "ILUVATAR",
               {true, true, true, true, true, true, true, true});
#endif

#if defined(INFINI_RT_TEST_WITH_HYGON)
  TestDispatch(&context, infini::rt::Device::Type::kHygon, "HYGON",
               {true, true, true, true, true, true, true, true});
#endif

#if defined(INFINI_RT_TEST_WITH_THEAD)
  TestDispatch(&context, infini::rt::Device::Type::kThead, "THEAD",
               {true, true, true, true, true, true, true, true});
#endif

#if defined(INFINI_RT_TEST_WITH_METAX)
  TestDispatch(&context, infini::rt::Device::Type::kMetax, "METAX",
               {true, true, true, true, true, true, true, true});
#endif

#if defined(INFINI_RT_TEST_WITH_MARS)
  TestDispatch(&context, infini::rt::Device::Type::kMars, "MARS",
               {true, true, true, true, true, true, true, true});
#endif

#if defined(INFINI_RT_TEST_WITH_MOORE)
  TestDispatch(&context, infini::rt::Device::Type::kMoore, "MOORE",
               {true, true, false, true, true, true, true, true});
#endif

#if defined(INFINI_RT_TEST_WITH_CAMBRICON)
  TestDispatch(&context, infini::rt::Device::Type::kCambricon, "CAMBRICON",
               {true, false, false, false, true, false, false, false});
#endif

#if defined(INFINI_RT_TEST_WITH_ASCEND)
  TestDispatch(&context, infini::rt::Device::Type::kAscend, "ASCEND",
               {true, false, false, false, true, false, false, false});
#endif

  return context.ExitCode();
}
