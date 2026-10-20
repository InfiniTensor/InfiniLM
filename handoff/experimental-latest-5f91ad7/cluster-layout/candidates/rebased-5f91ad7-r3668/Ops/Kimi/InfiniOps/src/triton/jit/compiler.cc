#include "triton/jit/compiler.h"

#include <pybind11/pybind11.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace py = pybind11;

namespace infini::ops::triton::jit::detail {
namespace {

bool PythonIsReady() {
  const bool ready = Py_IsInitialized() != 0;
  if (!ready) {
    std::fputs(
        "Triton JIT requires an initialized Python interpreter; use it "
        "through the InfiniOps Python package.\n",
        stderr);
  }

  return ready;
}

template <typename Result, typename Function>
std::optional<Result> RunPython(const char* operation, Function&& function) {
  if (!PythonIsReady()) return std::nullopt;

  try {
    py::gil_scoped_acquire gil;
    try {
      return std::forward<Function>(function)();
    } catch (const py::error_already_set& error) {
      std::fprintf(stderr, "Triton JIT %s failed:\n%s\n", operation,
                   error.what());
    } catch (const py::cast_error& error) {
      std::fprintf(stderr, "Triton JIT %s failed: %s\n", operation,
                   error.what());
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Triton JIT %s failed: %s\n", operation, error.what());
  }

  return std::nullopt;
}

py::object CompilerFunction(const char* name) {
  return py::module_::import("infini.triton.jit.compile").attr(name);
}

py::dict BuildTarget(const Target& target) {
  py::dict result;
  result["backend"] = target.backend;
  result["architecture"] = target.architecture;
  result["warp_size"] = target.warp_size;
  return result;
}

py::dict BuildConfig(const Config& config) {
  py::dict constexprs;
  for (const auto& [name, value] : config.constexprs()) {
    constexprs[py::str(name)] = value;
  }

  py::dict result;
  result["num_warps"] = config.num_warps();
  result["num_stages"] = config.num_stages();
  result["constexprs"] = std::move(constexprs);
  return result;
}

py::dict BuildCandidate(const AutoTuningCandidate& candidate) {
  py::dict result = BuildConfig(candidate.config);
  result["grid"] =
      py::make_tuple(candidate.grid.x, candidate.grid.y, candidate.grid.z);
  result["signature"] = candidate.signature;
  result["output_prefix"] = candidate.output_prefix;
  result["cache_identity"] = candidate.cache_identity;
  return result;
}

template <typename T>
T DecodeBits(std::uint64_t bits) {
  T value;
  static_assert(sizeof(value) <= sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

py::object BuildArgument(const Argument& argument) {
  switch (argument.type) {
    case DataType::kPointer:
    case DataType::kUInt64:
      return py::int_(argument.bits);
    case DataType::kInt8:
      return py::int_(DecodeBits<std::int8_t>(argument.bits));
    case DataType::kUInt8:
      return py::int_(DecodeBits<std::uint8_t>(argument.bits));
    case DataType::kInt16:
      return py::int_(DecodeBits<std::int16_t>(argument.bits));
    case DataType::kUInt16:
      return py::int_(DecodeBits<std::uint16_t>(argument.bits));
    case DataType::kInt32:
      return py::int_(DecodeBits<std::int32_t>(argument.bits));
    case DataType::kUInt32:
      return py::int_(DecodeBits<std::uint32_t>(argument.bits));
    case DataType::kInt64:
      return py::int_(DecodeBits<std::int64_t>(argument.bits));
    case DataType::kFloat32:
      return py::float_(DecodeBits<float>(argument.bits));
    case DataType::kFloat64:
      return py::float_(DecodeBits<double>(argument.bits));
  }

  assert(false && "Triton JIT received an unknown data type.");
  return py::none();
}

py::list BuildCandidates(const std::vector<AutoTuningCandidate>& candidates) {
  py::list result;
  for (const auto& candidate : candidates) {
    result.append(BuildCandidate(candidate));
  }
  return result;
}

py::list BuildArguments(const std::vector<Argument>& arguments) {
  py::list result;
  for (const auto& argument : arguments) {
    result.append(BuildArgument(argument));
  }
  return result;
}

}  // namespace

std::string CompilationFingerprint(const std::string& op_name) {
  static std::mutex cache_mutex;
  static std::unordered_map<std::string, std::string> cache;
  {
    const std::lock_guard<std::mutex> lock(cache_mutex);
    const auto cached = cache.find(op_name);
    if (cached != cache.end()) return cached->second;
  }

  auto value = RunPython<std::string>("fingerprint lookup", [&] {
    return CompilerFunction("get_compilation_fingerprint")(op_name)
        .cast<std::string>();
  });
  assert(value.has_value() &&
         "Triton JIT failed to get a compilation fingerprint.");
  if (!value.has_value()) return {};

  const std::lock_guard<std::mutex> lock(cache_mutex);
  return cache.try_emplace(op_name, std::move(*value)).first->second;
}

void Compile(const Target& target, const std::string& op_name,
             const std::string& output_prefix, const Config& config,
             const std::string& signature, const std::string& cache_identity) {
  const auto compiled = RunPython<bool>("compilation", [&] {
    CompilerFunction("compile_kernel")(op_name, output_prefix, signature,
                                       BuildConfig(config), BuildTarget(target),
                                       cache_identity);
    return true;
  });
  assert(compiled.has_value() && "Triton JIT compilation failed.");
}

std::optional<Config> AutoTune(
    const Target& target, int device_id, const std::string& op_name,
    const std::vector<AutoTuningCandidate>& candidates,
    const std::vector<Argument>& arguments, void* stream,
    int warmup_milliseconds, int repetition_milliseconds) {
  assert(!candidates.empty() && "Triton JIT requires tuning candidates.");
  assert(warmup_milliseconds >= 0 && repetition_milliseconds > 0 &&
         "Triton JIT auto-tuning durations are invalid.");
  if (candidates.empty() || warmup_milliseconds < 0 ||
      repetition_milliseconds <= 0) {
    return std::nullopt;
  }

  auto best_index = RunPython<long>("auto-tuning", [&] {
    return CompilerFunction("auto_tune")(
               op_name, BuildCandidates(candidates), BuildArguments(arguments),
               py::int_(reinterpret_cast<std::uintptr_t>(stream)),
               warmup_milliseconds, repetition_milliseconds,
               BuildTarget(target), device_id)
        .cast<long>();
  });
  assert(best_index.has_value() && "Triton JIT auto-tuning failed.");
  if (!best_index.has_value()) return std::nullopt;

  const bool valid_index =
      *best_index >= 0 &&
      static_cast<std::size_t>(*best_index) < candidates.size();
  assert(valid_index &&
         "Triton JIT auto-tuning returned an invalid candidate.");
  if (!valid_index) return std::nullopt;

  return candidates[static_cast<std::size_t>(*best_index)].config;
}

}  // namespace infini::ops::triton::jit::detail
