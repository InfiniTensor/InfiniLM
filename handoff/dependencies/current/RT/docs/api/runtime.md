# Runtime API

The runtime API is exposed through:

```cpp
#include <infini/rt.h>
```

Runtime functions live in `infini::rt::runtime`.

## Backend Selection

The top-level functions select which enabled backend receives
`infini::rt::runtime` calls:

```cpp
infini::rt::set_runtime_device_type(infini::rt::Device::Type::kCpu);
auto current = infini::rt::runtime_device_type();
```

The selected backend is thread-local. A GPU backend is selected initially when
one is enabled; otherwise CPU is selected.

Use the dispatching API for ordinary code:

```cpp
namespace rt = infini::rt::runtime;

infini::rt::set_runtime_device_type(infini::rt::Device::Type::kCpu);

void* ptr = nullptr;
rt::Malloc(&ptr, 1024);
rt::Free(ptr);
```

## CUDA Runtime API Alignment

The `infini::rt::runtime` namespace follows CUDA Runtime API naming, call
shapes, return-status conventions, and handle-oriented stream/event usage where
practical. Backend implementations translate those calls to the enabled runtime
backend, while unsupported operations return a non-success status without
changing the API shape. The signatures below are therefore listed without
per-function explanations; for CUDA-like operations, use the corresponding CUDA
Runtime API behavior as the baseline unless a backend support note says
otherwise.

## Runtime Status and Types

```cpp
runtime::Error
runtime::Stream
runtime::Event
runtime::Graph
runtime::GraphExec
runtime::StreamCaptureMode
runtime::MemcpyKind
runtime::kSuccess
runtime::kMemcpyHostToHost
runtime::kMemcpyHostToDevice
runtime::kMemcpyDeviceToHost
runtime::kMemcpyDeviceToDevice
```

`StreamCaptureMode` supports the same capture scopes used by CUDA-style stream
capture:

```cpp
runtime::StreamCaptureMode::kStreamCaptureModeGlobal
runtime::StreamCaptureMode::kStreamCaptureModeThreadLocal
runtime::StreamCaptureMode::kStreamCaptureModeRelaxed
```

## Device Functions

```cpp
runtime::Error SetDevice(int device);
runtime::Error GetDevice(int* device);
runtime::Error GetDeviceCount(int* count);
runtime::Error DeviceSynchronize();
```

## Memory Functions

```cpp
runtime::Error Malloc(void** ptr, std::size_t size);
runtime::Error MallocHost(void** ptr, std::size_t size);
runtime::Error MallocAsync(void** ptr, std::size_t size, runtime::Stream stream);
runtime::Error Free(void* ptr);
runtime::Error FreeHost(void* ptr);
runtime::Error FreeAsync(void* ptr, runtime::Stream stream);
runtime::Error MemGetInfo(std::size_t* free, std::size_t* total);
runtime::Error Memcpy(void* dst, const void* src, std::size_t count,
                      runtime::MemcpyKind kind);
runtime::Error MemcpyAsync(void* dst, const void* src, std::size_t count,
                           runtime::MemcpyKind kind, runtime::Stream stream);
runtime::Error Memset(void* ptr, int value, std::size_t count);
runtime::Error MemsetAsync(void* ptr, int value, std::size_t count,
                           runtime::Stream stream);
```

Not every backend supports every optional operation. Unsupported operations
return a non-success status. See [Backends](../backends.md).

## Stream and Event Functions

```cpp
runtime::Error StreamCreate(runtime::Stream* stream);
runtime::Error StreamDestroy(runtime::Stream stream);
runtime::Error StreamSynchronize(runtime::Stream stream);
runtime::Error StreamWaitEvent(runtime::Stream stream, runtime::Event event,
                               unsigned int flags);
runtime::Error EventCreate(runtime::Event* event);
runtime::Error EventCreateWithFlags(runtime::Event* event, unsigned int flags);
runtime::Error EventRecord(runtime::Event event, runtime::Stream stream);
runtime::Error EventQuery(runtime::Event event);
runtime::Error EventSynchronize(runtime::Event event);
runtime::Error EventDestroy(runtime::Event event);
runtime::Error EventElapsedTime(float* ms, runtime::Event start,
                                runtime::Event end);
```

## Graph Capture and Replay Functions

When the configured build includes a graph-capable backend, the generated
runtime API also includes graph capture and replay entry points.

The graph API captures stream work into a reusable `Graph`, instantiates it as a
launchable `GraphExec`, and replays it on a stream:

```cpp
runtime::Error StreamBeginCapture(runtime::Stream stream,
                                  runtime::StreamCaptureMode mode);
runtime::Error StreamEndCapture(runtime::Stream stream, runtime::Graph* graph);
runtime::Error GraphDestroy(runtime::Graph graph);
runtime::Error GraphInstantiate(runtime::GraphExec* graph_exec,
                                runtime::Graph graph);
runtime::Error GraphExecDestroy(runtime::GraphExec graph_exec);
runtime::Error GraphLaunch(runtime::GraphExec graph_exec,
                           runtime::Stream stream);
```

Use the same stream for `StreamBeginCapture` and `StreamEndCapture`. Operations
issued to that stream between the two calls become part of the captured graph.
After `GraphInstantiate` succeeds, the executable graph can be launched multiple
times while inputs and outputs remain at the recorded addresses.

```cpp
namespace rt = infini::rt::runtime;

rt::Stream stream = nullptr;
rt::Graph graph = nullptr;
rt::GraphExec graph_exec = nullptr;

rt::StreamCreate(&stream);

rt::StreamBeginCapture(
    stream, rt::StreamCaptureMode::kStreamCaptureModeRelaxed);
rt::MemcpyAsync(dst, src, count, rt::kMemcpyDeviceToDevice, stream);
rt::StreamEndCapture(stream, &graph);

rt::GraphInstantiate(&graph_exec, graph);
rt::GraphLaunch(graph_exec, stream);
rt::StreamSynchronize(stream);

rt::GraphExecDestroy(graph_exec);
rt::GraphDestroy(graph);
rt::StreamDestroy(stream);
```

Backends that do not support graph capture return a non-success status for the
graph entry points when those functions are present in the configured public
header. See [Backends](../backends.md) for current support.

## Backend-Specific Runtime Templates

Backend runtime specializations are available as implementation-facing headers,
for example:

```cpp
#include <infini/rt/cpu/runtime_.h>
```

These headers are useful for backend tests and low-level integrations, but
ordinary consumers should prefer the dispatching API from `<infini/rt.h>`.
