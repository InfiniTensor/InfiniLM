import argparse
import dataclasses
import pathlib
import re

_DETAIL_PREFIX = "infini/rt/detail"

_DEVICE_HEADERS = {
    "cpu": (
        ("cpu", "data_type_.h", "native/cpu/data_type_.h"),
        ("cpu", "device_.h", "native/cpu/device_.h"),
        ("cpu", "runtime_.h", "native/cpu/runtime_.h"),
    ),
    "nvidia": (
        ("nvidia", "data_type_.h", "native/cuda/nvidia/data_type_.h"),
        ("nvidia", "device_.h", "native/cuda/nvidia/device_.h"),
        ("nvidia", "runtime_.h", "native/cuda/nvidia/runtime_.h"),
        ("nvidia", "driver_.h", "native/cuda/nvidia/driver_.h"),
    ),
    "iluvatar": (
        ("iluvatar", "data_type_.h", "native/cuda/iluvatar/data_type_.h"),
        ("iluvatar", "device_.h", "native/cuda/iluvatar/device_.h"),
        ("iluvatar", "runtime_.h", "native/cuda/iluvatar/runtime_.h"),
    ),
    "hygon": (
        ("hygon", "data_type_.h", "native/cuda/hygon/data_type_.h"),
        ("hygon", "device_.h", "native/cuda/hygon/device_.h"),
        ("hygon", "runtime_.h", "native/cuda/hygon/runtime_.h"),
    ),
    "thead": (
        ("thead", "data_type_.h", "native/cuda/thead/data_type_.h"),
        ("thead", "device_.h", "native/cuda/thead/device_.h"),
        ("thead", "runtime_.h", "native/cuda/thead/runtime_.h"),
    ),
    "metax": (
        ("metax", "data_type_.h", "native/cuda/metax/data_type_.h"),
        ("metax", "device_.h", "native/cuda/metax/device_.h"),
        ("metax", "runtime_.h", "native/cuda/metax/runtime_.h"),
    ),
    "mars": (
        ("mars", "data_type_.h", "native/cuda/mars/data_type_.h"),
        ("mars", "device_.h", "native/cuda/mars/device_.h"),
        ("mars", "runtime_.h", "native/cuda/mars/runtime_.h"),
    ),
    "moore": (
        ("moore", "data_type_.h", "native/cuda/moore/data_type_.h"),
        ("moore", "device_.h", "native/cuda/moore/device_.h"),
        ("moore", "runtime_.h", "native/cuda/moore/runtime_.h"),
    ),
    "cambricon": (
        ("cambricon", "data_type_.h", "native/cambricon/data_type_.h"),
        ("cambricon", "device_.h", "native/cambricon/device_.h"),
        ("cambricon", "runtime_.h", "native/cambricon/runtime_.h"),
    ),
    "ascend": (
        ("ascend", "data_type_.h", "native/ascend/data_type_.h"),
        ("ascend", "device_.h", "native/ascend/device_.h"),
        ("ascend", "runtime_.h", "native/ascend/runtime_.h"),
    ),
}

_DEVICE_TYPES = {
    "cpu": "Device::Type::kCpu",
    "nvidia": "Device::Type::kNvidia",
    "iluvatar": "Device::Type::kIluvatar",
    "hygon": "Device::Type::kHygon",
    "thead": "Device::Type::kThead",
    "metax": "Device::Type::kMetax",
    "mars": "Device::Type::kMars",
    "moore": "Device::Type::kMoore",
    "cambricon": "Device::Type::kCambricon",
    "ascend": "Device::Type::kAscend",
}

_DEFAULT_DEVICE_PRIORITY = (
    "nvidia",
    "iluvatar",
    "hygon",
    "thead",
    "metax",
    "mars",
    "moore",
    "cambricon",
    "ascend",
    "cpu",
)


def _guard(path):
    token = "_".join(path.parts).replace(".", "_").upper()

    return f"INFINI_RT_GENERATED_{token}_"


def _write_wrapper(include_root, device, header_name, target):
    path = include_root / "infini" / "rt" / device / header_name
    guard = _guard(path.relative_to(include_root))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#ifndef {guard}
#define {guard}

#include <{_DETAIL_PREFIX}/{target}>

#endif
"""
    )


def _detail_include(path):
    return f"<{_DETAIL_PREFIX}/{path}>"


def _rewrite_detail_include(match):
    target = match.group(1)
    return f"#include {_detail_include(target)}"


_DETAIL_INCLUDE_PATTERN = re.compile(
    r'#include "((?:common|native)/[^"]+|data_type\.h|device\.h|dispatcher\.h|driver\.h|hash\.h|runtime\.h|tensor_view\.h)"'
)


def _detail_header_dependencies(source_root, relative_path):
    source_path = source_root / relative_path
    text = source_path.read_text()

    return {
        target
        for target in _DETAIL_INCLUDE_PATTERN.findall(text)
        if (source_root / target).exists()
    }


def _write_detail_header(include_root, source_root, relative_path):
    source_path = source_root / relative_path
    output_path = include_root / _DETAIL_PREFIX / relative_path
    text = source_path.read_text()
    text = _DETAIL_INCLUDE_PATTERN.sub(_rewrite_detail_include, text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)


def _write_detail_headers(include_root, source_root, devices):
    detail_headers = {
        "common/constexpr_map.h",
        "common/traits.h",
        "data_type.h",
        "device.h",
        "dispatcher.h",
        "driver.h",
        "hash.h",
        "runtime.h",
        "tensor_view.h",
    }

    for device in devices:
        for _, _, target in _DEVICE_HEADERS[device]:
            detail_headers.add(target)

    pending_headers = list(detail_headers)
    while pending_headers:
        relative_path = pending_headers.pop()
        for dependency in _detail_header_dependencies(source_root, relative_path):
            if dependency not in detail_headers:
                detail_headers.add(dependency)
                pending_headers.append(dependency)

    for relative_path in sorted(detail_headers):
        _write_detail_header(include_root, source_root, relative_path)


def _write_generated_header(include_root, source_root, devices):
    default_device = _default_device(devices)
    default_device_type = _DEVICE_TYPES[default_device]
    public_runtime_functions = _public_runtime_functions_for_devices(
        devices, source_root
    )
    has_graph_api = any(
        function.name in {"StreamBeginCapture", "GraphLaunch"}
        for function in public_runtime_functions
    )
    graph_declarations = (
        """
using Graph = void*;

using GraphExec = void*;

enum class StreamCaptureMode {
  kStreamCaptureModeGlobal = 0,
  kStreamCaptureModeThreadLocal = 1,
  kStreamCaptureModeRelaxed = 2,
};
"""
        if has_graph_api
        else ""
    )
    includes = [
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <type_traits>",
        f"#include {_detail_include('data_type.h')}",
        f"#include {_detail_include('device.h')}",
        f"#include {_detail_include('driver.h')}",
        f"#include {_detail_include('hash.h')}",
        f"#include {_detail_include('runtime.h')}",
        f"#include {_detail_include('tensor_view.h')}",
    ]

    for device in devices:
        includes.append(f"#include <infini/rt/{device}/device_.h>")

    for device in devices:
        includes.append(f"#include <infini/rt/{device}/runtime_.h>")

    for device in devices:
        if any(h == "driver_.h" for _, h, _ in _DEVICE_HEADERS[device]):
            includes.append(f"#include <infini/rt/{device}/driver_.h>")

    runtime_declarations = "\n\n".join(
        f"{function.signature()};" for function in public_runtime_functions
    )
    if any(function.name == "GetDeviceCount" for function in public_runtime_functions):
        runtime_declarations += (
            "\n\nError GetDeviceCount(Device::Type type, int* count);"
        )
    if any(function.name == "EventQuery" for function in public_runtime_functions):
        runtime_declarations += "\n\nbool IsEventNotReady(Error status);"

    path = include_root / "infini" / "rt" / "generated.h"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""#ifndef INFINI_RT_GENERATED_PUBLIC_H_
#define INFINI_RT_GENERATED_PUBLIC_H_

{chr(10).join(includes)}

namespace infini::rt {{

void set_runtime_device_type(Device::Type device_type);

Device::Type runtime_device_type();

namespace runtime {{
namespace generated_detail {{

using DefaultErrorRuntime = Runtime<{default_device_type}>;

inline constexpr Device::Type kDefaultDeviceType = {default_device_type};

}}  // namespace generated_detail

using Error = typename generated_detail::DefaultErrorRuntime::Error;

using Stream = typename generated_detail::DefaultErrorRuntime::Stream;

using Event = void*;

{graph_declarations}
using MemcpyKind = std::remove_cv_t<
    decltype(generated_detail::DefaultErrorRuntime::kMemcpyHostToHost)>;

inline constexpr Error kSuccess = generated_detail::DefaultErrorRuntime::kSuccess;

inline constexpr MemcpyKind kMemcpyHostToHost =
    generated_detail::DefaultErrorRuntime::kMemcpyHostToHost;

inline constexpr MemcpyKind kMemcpyHostToDevice =
    generated_detail::DefaultErrorRuntime::kMemcpyHostToDevice;

inline constexpr MemcpyKind kMemcpyDeviceToHost =
    generated_detail::DefaultErrorRuntime::kMemcpyDeviceToHost;

inline constexpr MemcpyKind kMemcpyDeviceToDevice =
    generated_detail::DefaultErrorRuntime::kMemcpyDeviceToDevice;

{runtime_declarations}

}}  // namespace runtime
}}  // namespace infini::rt

#endif
"""
    )


@dataclasses.dataclass(frozen=True)
class _Param:
    type: str
    name: str


@dataclasses.dataclass(frozen=True)
class _Function:
    return_type: str
    name: str
    params: tuple[_Param, ...]

    def signature(self):
        return f"{self.return_type} {self.name}({self.params_decl()})"

    def params_decl(self):
        return ", ".join(f"{param.type} {param.name}" for param in self.params)


_PUBLIC_RUNTIME_FUNCTIONS = (
    _Function("Error", "SetDevice", (_Param("int", "device"),)),
    _Function("Error", "GetDevice", (_Param("int*", "device"),)),
    _Function("Error", "GetDeviceCount", (_Param("int*", "count"),)),
    _Function("Error", "DeviceSynchronize", ()),
    _Function(
        "Error",
        "Malloc",
        (_Param("void**", "ptr"), _Param("std::size_t", "size")),
    ),
    _Function(
        "Error",
        "MallocHost",
        (_Param("void**", "ptr"), _Param("std::size_t", "size")),
    ),
    _Function(
        "Error",
        "MallocAsync",
        (
            _Param("void**", "ptr"),
            _Param("std::size_t", "size"),
            _Param("Stream", "stream"),
        ),
    ),
    _Function("Error", "Free", (_Param("void*", "ptr"),)),
    _Function("Error", "FreeHost", (_Param("void*", "ptr"),)),
    _Function(
        "Error",
        "FreeAsync",
        (_Param("void*", "ptr"), _Param("Stream", "stream")),
    ),
    _Function(
        "Error",
        "MemGetInfo",
        (_Param("std::size_t*", "free"), _Param("std::size_t*", "total")),
    ),
    _Function(
        "Error",
        "Memcpy",
        (
            _Param("void*", "dst"),
            _Param("const void*", "src"),
            _Param("std::size_t", "count"),
            _Param("MemcpyKind", "kind"),
        ),
    ),
    _Function(
        "Error",
        "MemcpyAsync",
        (
            _Param("void*", "dst"),
            _Param("const void*", "src"),
            _Param("std::size_t", "count"),
            _Param("MemcpyKind", "kind"),
            _Param("Stream", "stream"),
        ),
    ),
    _Function(
        "Error",
        "Memset",
        (
            _Param("void*", "ptr"),
            _Param("int", "value"),
            _Param("std::size_t", "count"),
        ),
    ),
    _Function(
        "Error",
        "MemsetAsync",
        (
            _Param("void*", "ptr"),
            _Param("int", "value"),
            _Param("std::size_t", "count"),
            _Param("Stream", "stream"),
        ),
    ),
    _Function("Error", "StreamCreate", (_Param("Stream*", "stream"),)),
    _Function("Error", "StreamDestroy", (_Param("Stream", "stream"),)),
    _Function("Error", "StreamSynchronize", (_Param("Stream", "stream"),)),
    _Function(
        "Error",
        "StreamWaitEvent",
        (
            _Param("Stream", "stream"),
            _Param("Event", "event"),
            _Param("unsigned int", "flags"),
        ),
    ),
    _Function("Error", "EventCreate", (_Param("Event*", "event"),)),
    _Function(
        "Error",
        "EventCreateWithFlags",
        (_Param("Event*", "event"), _Param("unsigned int", "flags")),
    ),
    _Function(
        "Error",
        "EventRecord",
        (_Param("Event", "event"), _Param("Stream", "stream")),
    ),
    _Function("Error", "EventQuery", (_Param("Event", "event"),)),
    _Function("Error", "EventSynchronize", (_Param("Event", "event"),)),
    _Function("Error", "EventDestroy", (_Param("Event", "event"),)),
    _Function(
        "Error",
        "EventElapsedTime",
        (
            _Param("float*", "ms"),
            _Param("Event", "start"),
            _Param("Event", "end"),
        ),
    ),
    _Function(
        "Error",
        "StreamBeginCapture",
        (_Param("Stream", "stream"), _Param("StreamCaptureMode", "mode")),
    ),
    _Function(
        "Error",
        "StreamEndCapture",
        (_Param("Stream", "stream"), _Param("Graph*", "graph")),
    ),
    _Function("Error", "GraphDestroy", (_Param("Graph", "graph"),)),
    _Function(
        "Error",
        "GraphInstantiate",
        (_Param("GraphExec*", "graph_exec"), _Param("Graph", "graph")),
    ),
    _Function(
        "Error",
        "GraphExecDestroy",
        (_Param("GraphExec", "graph_exec"),),
    ),
    _Function(
        "Error",
        "GraphLaunch",
        (_Param("GraphExec", "graph_exec"), _Param("Stream", "stream")),
    ),
)


def _default_device(devices):
    for device in _DEFAULT_DEVICE_PRIORITY:
        if device in devices:
            return device

    raise ValueError("at least one device is required")


def _runtime_arg(param, device):
    device_type = _DEVICE_TYPES[device]
    if param.type == "MemcpyKind":
        return f"RuntimeMemcpyKind<{device_type}>({param.name})"
    if param.type == "Stream":
        return (
            f"reinterpret_cast<typename Runtime<{device_type}>::Stream>({param.name})"
        )
    if param.type == "Stream*":
        return (
            f"reinterpret_cast<typename Runtime<{device_type}>::Stream*>({param.name})"
        )
    if param.type == "Event":
        return f"reinterpret_cast<typename Runtime<{device_type}>::Event>({param.name})"
    if param.type == "Event*":
        return (
            f"reinterpret_cast<typename Runtime<{device_type}>::Event*>({param.name})"
        )
    if param.type == "Graph":
        return f"reinterpret_cast<typename Runtime<{device_type}>::Graph>({param.name})"
    if param.type == "Graph*":
        return (
            f"reinterpret_cast<typename Runtime<{device_type}>::Graph*>({param.name})"
        )
    if param.type == "GraphExec":
        return (
            f"reinterpret_cast<typename Runtime<{device_type}>::GraphExec>"
            f"({param.name})"
        )
    if param.type == "GraphExec*":
        return (
            f"reinterpret_cast<typename Runtime<{device_type}>::GraphExec*>"
            f"({param.name})"
        )
    if param.type == "StreamCaptureMode":
        return f"RuntimeStreamCaptureMode<{device_type}>({param.name})"

    return param.name


def _runtime_args(function, device):
    args = (_runtime_arg(param, device) for param in function.params)

    return ", ".join(arg for arg in args if arg is not None)


def _runtime_call(function, device):
    device_type = _DEVICE_TYPES[device]
    args = _runtime_args(function, device)
    if args:
        return f"Runtime<{device_type}>::{function.name}({args})"

    return f"Runtime<{device_type}>::{function.name}()"


def _dispatch_cases(devices, function):
    return "\n".join(
        f"""    case {_DEVICE_TYPES[device]}:
      return CheckCall([&] {{ return {_runtime_call(function, device)}; }});"""
        for device in devices
    )


def _write_runtime_dispatch_function(function, devices):
    return f"""{function.signature()} {{
  switch (infini::rt::runtime_device_type()) {{
{_dispatch_cases(devices, function)}
  }}

  assert(false && "unsupported runtime device type");
  return InvalidValueError();
}}
"""


def _write_explicit_device_count_query(devices):
    cases = "\n".join(
        f"    case {_DEVICE_TYPES[device]}:\n"
        f"      return QueryDeviceCount<{_DEVICE_TYPES[device]}>(count);"
        for device in devices
    )
    return f"""namespace {{

template <Device::Type type>
Error QueryDeviceCount(int* count) {{
  using DeviceRuntime = Runtime<type>;
  int queried_count = -1;
  const auto status = DeviceRuntime::GetDeviceCount(&queried_count);
  if (status != DeviceRuntime::kSuccess) return static_cast<Error>(status);
  if (queried_count < 0) return InvalidValueError();
  *count = queried_count;
  return kSuccess;
}}

}}  // namespace

Error GetDeviceCount(Device::Type type, int* count) {{
  if (count == nullptr) return InvalidValueError();
  switch (type) {{
{cases}
    default:
      return InvalidValueError();
  }}
}}
"""


def _runtime_header_for_device(source_root, device):
    for _, header_name, target in _DEVICE_HEADERS[device]:
        if header_name == "runtime_.h":
            return source_root / target

    raise ValueError(f"device {device!r} does not have a runtime header")


def _devices_for_function(function, devices, source_root):
    pattern = re.compile(r"\b" + re.escape(function.name) + r"\b")

    return tuple(
        device
        for device in devices
        if pattern.search(_runtime_header_for_device(source_root, device).read_text())
    )


def _public_runtime_functions_for_devices(devices, source_root):
    return tuple(
        function
        for function in _PUBLIC_RUNTIME_FUNCTIONS
        if _devices_for_function(function, devices, source_root)
    )


def _write_runtime_dispatch(source_path, source_root, devices):
    functions = _public_runtime_functions_for_devices(devices, source_root)
    event_not_ready_dispatch = ""
    event_query = next(
        (function for function in functions if function.name == "EventQuery"), None
    )
    if event_query is not None:
        event_not_ready_cases = "\n".join(
            f"    case {_DEVICE_TYPES[device]}:\n"
            f"      return status == static_cast<Error>(Runtime<{_DEVICE_TYPES[device]}>::kEventNotReady);"
            for device in _devices_for_function(event_query, devices, source_root)
            if re.search(
                r"\bkEventNotReady\b",
                _runtime_header_for_device(source_root, device).read_text(),
            )
        )
        event_not_ready_dispatch = f"""
bool IsEventNotReady(Error status) {{
  if (status == kSuccess) return false;
  switch (infini::rt::runtime_device_type()) {{
{event_not_ready_cases}
    default:
      return false;
  }}
}}
"""
    stream_capture_mode_helper = (
        """
template <Device::Type device_type>
auto RuntimeStreamCaptureMode(StreamCaptureMode mode) {
  using DeviceRuntime = Runtime<device_type>;

  switch (mode) {
    case StreamCaptureMode::kStreamCaptureModeGlobal:
      return DeviceRuntime::kStreamCaptureModeGlobal;
    case StreamCaptureMode::kStreamCaptureModeThreadLocal:
      return DeviceRuntime::kStreamCaptureModeThreadLocal;
    case StreamCaptureMode::kStreamCaptureModeRelaxed:
      return DeviceRuntime::kStreamCaptureModeRelaxed;
  }

  assert(false && "unsupported stream capture mode");
  return DeviceRuntime::kStreamCaptureModeRelaxed;
}
"""
        if any(
            param.type == "StreamCaptureMode"
            for function in functions
            for param in function.params
        )
        else ""
    )
    dispatch_functions = "\n".join(
        _write_runtime_dispatch_function(
            function,
            devices=_devices_for_function(function, devices, source_root),
        )
        for function in functions
    )
    device_count_query = next(
        (function for function in functions if function.name == "GetDeviceCount"), None
    )
    explicit_device_count_query = (
        _write_explicit_device_count_query(
            _devices_for_function(device_count_query, devices, source_root)
        )
        if device_count_query is not None
        else ""
    )
    set_device_type_cases = "\n".join(
        f"""    case {_DEVICE_TYPES[device]}:
      runtime_device_type_ = device_type;
      return;"""
        for device in devices
    )

    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        f"""#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <infini/rt/generated.h>

namespace infini::rt {{
namespace {{

thread_local Device::Type runtime_device_type_ =
    runtime::generated_detail::kDefaultDeviceType;

}}  // namespace

void set_runtime_device_type(Device::Type device_type) {{
  switch (device_type) {{
{set_device_type_cases}
  }}

  assert(false && "unsupported runtime device type");
}}

Device::Type runtime_device_type() {{
  return runtime_device_type_;
}}

}}  // namespace infini::rt

namespace infini::rt::runtime {{
namespace {{

template <typename Func>
Error CheckCall(Func&& func) {{
  using ReturnType = decltype(std::forward<Func>(func)());

  if constexpr (std::is_void_v<ReturnType>) {{
    std::forward<Func>(func)();
    return kSuccess;
  }} else {{
    return static_cast<Error>(std::forward<Func>(func)());
  }}
}}

Error InvalidValueError() {{ return static_cast<Error>(1); }}

template <Device::Type device_type>
auto RuntimeMemcpyKind(MemcpyKind kind) {{
  using DeviceRuntime = Runtime<device_type>;

  switch (kind) {{
    case kMemcpyHostToHost:
      return DeviceRuntime::kMemcpyHostToHost;
    case kMemcpyHostToDevice:
      return DeviceRuntime::kMemcpyHostToDevice;
    case kMemcpyDeviceToHost:
      return DeviceRuntime::kMemcpyDeviceToHost;
    case kMemcpyDeviceToDevice:
      return DeviceRuntime::kMemcpyDeviceToDevice;
  }}

  assert(false && "unsupported memcpy kind");
  return DeviceRuntime::kMemcpyHostToHost;
}}

{stream_capture_mode_helper}
}}  // namespace

{dispatch_functions}
{explicit_device_count_query}
{event_not_ready_dispatch}
}}  // namespace infini::rt::runtime
"""
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="generated/include")
    parser.add_argument("--source-output", default="generated/src/runtime_dispatch.cc")
    parser.add_argument("--runtime-header", default="src/runtime.h")
    parser.add_argument("--devices", nargs="+", required=True)
    args = parser.parse_args()

    devices = []
    for device in args.devices:
        if device not in _DEVICE_HEADERS:
            raise ValueError(f"unsupported device {device!r}")
        if device not in devices:
            devices.append(device)

    include_root = pathlib.Path(args.output_dir)
    source_root = pathlib.Path(args.runtime_header).parent

    _write_detail_headers(include_root, source_root, devices)
    for device in devices:
        for wrapper_device, header_name, target in _DEVICE_HEADERS[device]:
            _write_wrapper(include_root, wrapper_device, header_name, target)

    _write_generated_header(include_root, source_root, devices)
    _write_runtime_dispatch(pathlib.Path(args.source_output), source_root, devices)


if __name__ == "__main__":
    main()
