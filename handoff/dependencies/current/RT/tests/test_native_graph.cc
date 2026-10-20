#include <infini/rt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string_view>

#include "test_helper.h"

namespace {

namespace runtime = infini::rt::runtime;

void ExpectSuccess(infini::rt::test::TestContext* context,
                   runtime::Error status, std::string_view message) {
  context->Expect(status == runtime::kSuccess, message);
}

void FillPattern(std::array<std::uint8_t, 16>* input, std::uint8_t salt) {
  for (std::size_t i = 0; i < input->size(); ++i) {
    (*input)[i] = static_cast<std::uint8_t>(i * 13 + salt);
  }
}

bool CopyDeviceToHostAndValidate(infini::rt::test::TestContext* context,
                                 void* device_ptr,
                                 const std::array<std::uint8_t, 16>& expected,
                                 std::string_view message) {
  std::array<std::uint8_t, 16> output{};
  ExpectSuccess(context,
                runtime::Memcpy(output.data(), device_ptr, output.size(),
                                runtime::kMemcpyDeviceToHost),
                "Failed to copy graph output to host.");
  return context->ExpectEqual(output, expected, message);
}

bool SelectTestDevice(infini::rt::test::TestContext* context) {
  infini::rt::set_runtime_device_type(INFINI_RT_TEST_DEVICE_TYPE);

  int count = 0;
  ExpectSuccess(context, runtime::GetDeviceCount(&count),
                "Failed to query device count.");
  if (context->ExitCode() != 0) {
    return false;
  }

  if (count == 0) {
    std::cout << "Skipping " << INFINI_RT_TEST_BACKEND_NAME
              << " graph test because no device is available." << std::endl;
    return false;
  }

  ExpectSuccess(context, runtime::SetDevice(0),
                "Failed to set graph test device.");
  return context->ExitCode() == 0;
}

void RunUnsupportedGraphSmoke(infini::rt::test::TestContext* context) {
  runtime::Stream stream = nullptr;
  ExpectSuccess(context, runtime::StreamCreate(&stream),
                "Failed to create stream for unsupported graph smoke.");
  if (context->ExitCode() != 0) {
    return;
  }

  const auto status = runtime::StreamBeginCapture(
      stream, runtime::StreamCaptureMode::kStreamCaptureModeRelaxed);
  context->Expect(status != runtime::kSuccess,
                  "Unsupported graph capture should return an error.");

  ExpectSuccess(context, runtime::StreamDestroy(stream),
                "Failed to destroy stream for unsupported graph smoke.");
}

void RunGraphReplayTest(infini::rt::test::TestContext* context) {
  void* src = nullptr;
  void* dst = nullptr;
  runtime::Stream stream = nullptr;
  runtime::Graph graph = nullptr;
  runtime::GraphExec graph_exec = nullptr;

  std::array<std::uint8_t, 16> capture_input{};
  FillPattern(&capture_input, 7);

  ExpectSuccess(context, runtime::Malloc(&src, capture_input.size()),
                "Failed to allocate source buffer.");
  ExpectSuccess(context, runtime::Malloc(&dst, capture_input.size()),
                "Failed to allocate destination buffer.");
  ExpectSuccess(context, runtime::StreamCreate(&stream),
                "Failed to create stream.");

  if (context->ExitCode() == 0) {
    ExpectSuccess(
        context,
        runtime::Memcpy(src, capture_input.data(), capture_input.size(),
                        runtime::kMemcpyHostToDevice),
        "Failed to initialize source buffer.");
    ExpectSuccess(context, runtime::Memset(dst, 0, capture_input.size()),
                  "Failed to initialize destination buffer.");

    ExpectSuccess(
        context,
        runtime::StreamBeginCapture(
            stream, runtime::StreamCaptureMode::kStreamCaptureModeRelaxed),
        "Failed to begin stream capture.");
    ExpectSuccess(context,
                  runtime::MemcpyAsync(dst, src, capture_input.size(),
                                       runtime::kMemcpyDeviceToDevice, stream),
                  "Failed to record device-to-device copy.");
    ExpectSuccess(context, runtime::StreamEndCapture(stream, &graph),
                  "Failed to end stream capture.");
    context->Expect(graph != nullptr,
                    "Stream capture should produce a graph handle.");

    ExpectSuccess(context, runtime::GraphInstantiate(&graph_exec, graph),
                  "Failed to instantiate graph.");
    context->Expect(graph_exec != nullptr,
                    "Graph instantiation should produce an executable handle.");
  }

  std::array<std::uint8_t, 16> replay_input_1{};
  std::array<std::uint8_t, 16> replay_input_2{};
  FillPattern(&replay_input_1, 31);
  FillPattern(&replay_input_2, 53);

  if (context->ExitCode() == 0) {
    ExpectSuccess(
        context,
        runtime::Memcpy(src, replay_input_1.data(), replay_input_1.size(),
                        runtime::kMemcpyHostToDevice),
        "Failed to refresh first source buffer.");
    ExpectSuccess(context, runtime::Memset(dst, 0, replay_input_1.size()),
                  "Failed to clear first destination buffer.");
    ExpectSuccess(context, runtime::DeviceSynchronize(),
                  "Failed to synchronize first replay inputs.");
    ExpectSuccess(context, runtime::GraphLaunch(graph_exec, stream),
                  "Failed to launch first graph replay.");
    ExpectSuccess(context, runtime::StreamSynchronize(stream),
                  "Failed to synchronize first graph replay.");
    CopyDeviceToHostAndValidate(context, dst, replay_input_1,
                                "First graph replay should copy D2D data.");
  }

  if (context->ExitCode() == 0) {
    ExpectSuccess(
        context,
        runtime::Memcpy(src, replay_input_2.data(), replay_input_2.size(),
                        runtime::kMemcpyHostToDevice),
        "Failed to refresh second source buffer.");
    ExpectSuccess(context, runtime::Memset(dst, 0, replay_input_2.size()),
                  "Failed to clear second destination buffer.");
    ExpectSuccess(context, runtime::DeviceSynchronize(),
                  "Failed to synchronize second replay inputs.");
    ExpectSuccess(context, runtime::GraphLaunch(graph_exec, stream),
                  "Failed to launch second graph replay.");
    ExpectSuccess(context, runtime::StreamSynchronize(stream),
                  "Failed to synchronize second graph replay.");
    CopyDeviceToHostAndValidate(context, dst, replay_input_2,
                                "Second graph replay should copy D2D data.");
  }

  if (graph_exec != nullptr) {
    ExpectSuccess(context, runtime::GraphExecDestroy(graph_exec),
                  "Failed to destroy graph exec.");
  }
  if (graph != nullptr) {
    ExpectSuccess(context, runtime::GraphDestroy(graph),
                  "Failed to destroy graph.");
  }
  if (stream != nullptr) {
    ExpectSuccess(context, runtime::StreamDestroy(stream),
                  "Failed to destroy stream.");
  }
  if (dst != nullptr) {
    ExpectSuccess(context, runtime::Free(dst),
                  "Failed to free destination buffer.");
  }
  if (src != nullptr) {
    ExpectSuccess(context, runtime::Free(src), "Failed to free source buffer.");
  }
}

}  // namespace

int main() {
  infini::rt::test::TestContext context;

  if (!SelectTestDevice(&context)) {
    return context.ExitCode();
  }

  if constexpr (INFINI_RT_TEST_SUPPORTS_GRAPH_CAPTURE) {
    RunGraphReplayTest(&context);
  } else {
    RunUnsupportedGraphSmoke(&context);
  }

  return context.ExitCode();
}
