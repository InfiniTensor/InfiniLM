#ifndef INFINI_OPS_TRITON_JIT_H_
#define INFINI_OPS_TRITON_JIT_H_

#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "data_type.h"
#include "tensor.h"
#include "triton/jit/backend.h"
#include "triton/jit/cache.h"
#include "triton/jit/compiler.h"
#include "triton/jit/config_.h"

namespace infini::ops::triton::jit {

namespace detail {

struct ScalarTypeDescriptor {
  const char* name;

  DataType data_type;
};

inline const char* TritonTypeName(::infini::ops::DataType data_type) {
  switch (data_type) {
    case ::infini::ops::DataType::kFloat16:
      return "fp16";
    case ::infini::ops::DataType::kBFloat16:
      return "bf16";
    case ::infini::ops::DataType::kFloat32:
      return "fp32";
    case ::infini::ops::DataType::kFloat64:
      return "fp64";
    case ::infini::ops::DataType::kInt8:
      return "i8";
    case ::infini::ops::DataType::kInt16:
      return "i16";
    case ::infini::ops::DataType::kInt32:
      return "i32";
    case ::infini::ops::DataType::kInt64:
      return "i64";
    case ::infini::ops::DataType::kUInt8:
      return "u8";
    case ::infini::ops::DataType::kUInt16:
      return "u16";
    case ::infini::ops::DataType::kUInt32:
      return "u32";
    case ::infini::ops::DataType::kUInt64:
      return "u64";
  }
  return nullptr;
}

template <typename T>
constexpr ScalarTypeDescriptor ScalarDescriptor() {
  using Value = std::remove_cv_t<T>;
  static_assert(
      std::is_integral_v<Value> && (sizeof(Value) == 1 || sizeof(Value) == 2 ||
                                    sizeof(Value) == 4 || sizeof(Value) == 8),
      "Triton JIT does not support this scalar argument type.");
  if constexpr (std::is_same_v<Value, bool>) {
    return {"i32", DataType::kInt32};
  } else if constexpr (std::is_integral_v<Value> && sizeof(Value) == 1) {
    return std::is_signed_v<Value>
               ? ScalarTypeDescriptor{"i8", DataType::kInt8}
               : ScalarTypeDescriptor{"u8", DataType::kUInt8};
  } else if constexpr (std::is_integral_v<Value> && sizeof(Value) == 2) {
    return std::is_signed_v<Value>
               ? ScalarTypeDescriptor{"i16", DataType::kInt16}
               : ScalarTypeDescriptor{"u16", DataType::kUInt16};
  } else if constexpr (std::is_integral_v<Value> && sizeof(Value) == 4) {
    return std::is_signed_v<Value>
               ? ScalarTypeDescriptor{"i32", DataType::kInt32}
               : ScalarTypeDescriptor{"u32", DataType::kUInt32};
  } else {
    return std::is_signed_v<Value>
               ? ScalarTypeDescriptor{"i64", DataType::kInt64}
               : ScalarTypeDescriptor{"u64", DataType::kUInt64};
  }
}

template <typename T>
std::uint64_t ArgumentBits(const T& value) {
  static_assert(sizeof(T) <= sizeof(std::uint64_t),
                "`Triton` scalar argument is wider than `uint64_t`");
  std::uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(T));
  return bits;
}

class ArgumentPack {
 public:
  void Push(const Tensor& tensor) {
    const char* type_name = TritonTypeName(tensor.dtype());
    assert(type_name != nullptr &&
           "Triton JIT does not support this tensor type.");
    if (type_name == nullptr) {
      valid_ = false;
      return;
    }

    const auto data_ptr = reinterpret_cast<std::uintptr_t>(tensor.data());
    signature_ += "*" + std::string{type_name};
    if (data_ptr % 16 == 0) signature_ += ":16";
    signature_ += ",";

    const auto bits = static_cast<std::uint64_t>(data_ptr);
    arguments_.push_back({DataType::kPointer, bits});
    launch_arguments_.push_back(Store(bits));
  }

  template <typename T,
            std::enable_if_t<std::is_integral_v<std::remove_cv_t<T>>, int> = 0>
  void Push(T value) {
    constexpr auto descriptor = ScalarDescriptor<T>();
    signature_ += descriptor.name;

    if constexpr (std::is_same_v<std::remove_cv_t<T>, bool>) {
      signature_ += ",";
      const std::int32_t normalized = value;
      const auto bits = ArgumentBits(normalized);
      arguments_.push_back({descriptor.data_type, bits});
      launch_arguments_.push_back(Store(normalized));
    } else {
      bool compile_time_one = false;
      if (value == 1) {
        signature_ += ":1";
        compile_time_one = true;
      } else if ((value & 15) == 0) {
        signature_ += ":16";
      }
      signature_ += ",";

      const auto bits = ArgumentBits(value);
      arguments_.push_back({descriptor.data_type, bits});
      if (!compile_time_one) launch_arguments_.push_back(Store(value));
    }
  }

  void Push(float value) {
    signature_ += "fp32,";
    const auto bits = ArgumentBits(value);
    arguments_.push_back({DataType::kFloat32, bits});
    launch_arguments_.push_back(Store(value));
  }

  void Push(double value) {
    signature_ += "fp64,";
    const auto bits = ArgumentBits(value);
    arguments_.push_back({DataType::kFloat64, bits});
    launch_arguments_.push_back(Store(value));
  }

  std::string RuntimeSignature() const {
    std::string signature = signature_;
    if (!signature.empty()) signature.pop_back();
    return signature;
  }

  bool valid() const { return valid_; }

  const std::vector<Argument>& arguments() const { return arguments_; }

  void AddScratchArguments() {
    void* scratch_ptr = Store<std::uint64_t>(0);
    launch_arguments_.push_back(scratch_ptr);
    launch_arguments_.push_back(scratch_ptr);
  }

  void** launch_arguments() { return launch_arguments_.data(); }

 private:
  template <typename T>
  void* Store(const T& value) {
    storage_.push_back(ArgumentBits(value));
    return &storage_.back();
  }

  std::deque<std::uint64_t> storage_;

  std::vector<void*> launch_arguments_;

  std::vector<Argument> arguments_;

  std::string signature_;

  bool valid_{true};
};

template <Device::Type kDev>
const Kernel<kDev>* GetOrLoadKernel(const Target& target, int device_id,
                                    const std::string& compilation_fingerprint,
                                    const std::string& operator_name,
                                    const std::string& signature,
                                    const Config& config) {
  using CurrentBackend = Backend<kDev>;
  auto key = KernelCacheKey::Build(target, device_id, compilation_fingerprint,
                                   operator_name, signature, config);
  auto& cache = KernelCache<kDev>::Instance();
  if (const auto* kernel_ptr = cache.Find(key)) return kernel_ptr;

  const std::string output_prefix = key.ArtifactPrefix();
  assert(!output_prefix.empty() && "Triton JIT cache path is unavailable.");
  if (output_prefix.empty()) return nullptr;

  auto artifact = ReadKernelArtifact(output_prefix, key.identity());
  if (!artifact.has_value()) {
    Compile(target, operator_name, output_prefix, config, signature,
            key.identity());
    artifact = ReadKernelArtifact(output_prefix, key.identity());
  }
  assert(artifact.has_value() &&
         "Triton JIT did not produce a kernel artifact.");
  if (!artifact.has_value()) return nullptr;

  const bool scratch_memory_supported =
      artifact->metadata.global_scratch_size == 0 &&
      artifact->metadata.profile_scratch_size == 0;
  assert(scratch_memory_supported &&
         "Triton JIT scratch memory is not supported.");
  if (!scratch_memory_supported) return nullptr;

  auto kernel_ptr =
      CurrentBackend::LoadKernel(artifact->binary, artifact->metadata);
  assert(kernel_ptr != nullptr && "Triton JIT failed to load a kernel.");
  if (kernel_ptr == nullptr) return nullptr;

  return cache.InsertOrGet(key, std::move(kernel_ptr));
}

template <Device::Type kDev, typename GridFunction>
void LaunchConfigured(const Target& target, int device_id,
                      const std::string& compilation_fingerprint,
                      const std::string& operator_name, void* stream,
                      const Config& config, GridFunction& grid_function,
                      ArgumentPack& arguments) {
  using CurrentBackend = Backend<kDev>;
  const auto grid = grid_function(config);
  assert(grid.has_value() && "Triton JIT grid is invalid.");
  if (!grid.has_value()) return;

  const std::string signature = arguments.RuntimeSignature();
  const auto* kernel_ptr =
      GetOrLoadKernel<kDev>(target, device_id, compilation_fingerprint,
                            operator_name, signature, config);
  assert(kernel_ptr != nullptr && "Triton JIT failed to load a kernel.");
  if (kernel_ptr == nullptr) return;

  arguments.AddScratchArguments();
  CurrentBackend::Launch(*kernel_ptr, *grid, target.warp_size, stream,
                         arguments.launch_arguments());
}

template <Device::Type kDev, typename GridFunction>
void LaunchAutoTuned(const Target& target, int device_id,
                     const std::string& compilation_fingerprint,
                     const std::string& operator_name, void* stream,
                     const AutoTuningOptions& options,
                     const std::vector<std::string>& key_names,
                     const std::vector<std::uint64_t>& key_values,
                     GridFunction& grid_function, ArgumentPack& arguments) {
  using CurrentBackend = Backend<kDev>;
  std::vector<Grid> grids;
  grids.reserve(options.candidates.size());
  for (const auto& candidate : options.candidates) {
    const auto grid = grid_function(candidate);
    assert(grid.has_value() && "Triton JIT grid is invalid.");
    if (!grid.has_value()) return;
    grids.push_back(*grid);
  }

  const std::string signature = arguments.RuntimeSignature();
  const auto auto_tuning_key = AutoTuningCacheKey::Build(
      target, compilation_fingerprint, operator_name, signature, key_names,
      key_values, options.candidates, grids, options.warmup_milliseconds,
      options.repetition_milliseconds);

  auto& auto_tuning_cache = AutoTuningCache::Instance();
  auto best_config = auto_tuning_cache.Find(auto_tuning_key);
  if (!best_config.has_value()) {
    std::vector<AutoTuningCandidate> candidates;
    candidates.reserve(options.candidates.size());
    for (std::size_t index = 0; index < options.candidates.size(); ++index) {
      const auto& config = options.candidates[index];
      const auto kernel_key =
          KernelCacheKey::Build(target, device_id, compilation_fingerprint,
                                operator_name, signature, config);
      const std::string output_prefix = kernel_key.ArtifactPrefix();
      assert(!output_prefix.empty() && "Triton JIT cache path is unavailable.");
      if (output_prefix.empty()) return;
      candidates.push_back({config, grids[index], signature, output_prefix,
                            kernel_key.identity()});
    }

    auto tuned_config = AutoTune(
        target, device_id, operator_name, candidates, arguments.arguments(),
        stream, options.warmup_milliseconds, options.repetition_milliseconds);
    assert(tuned_config.has_value() && "Triton JIT auto-tuning failed.");
    if (!tuned_config.has_value()) return;
    best_config = std::move(tuned_config);
    auto_tuning_cache.Insert(auto_tuning_key, *best_config);
  }

  const auto grid = grid_function(*best_config);
  assert(grid.has_value() && "Triton JIT grid is invalid.");
  if (!grid.has_value()) return;
  const auto* kernel_ptr =
      GetOrLoadKernel<kDev>(target, device_id, compilation_fingerprint,
                            operator_name, signature, *best_config);
  assert(kernel_ptr != nullptr && "Triton JIT failed to load a kernel.");
  if (kernel_ptr == nullptr) return;

  arguments.AddScratchArguments();
  CurrentBackend::Launch(*kernel_ptr, *grid, target.warp_size, stream,
                         arguments.launch_arguments());
}

}  // namespace detail

template <Device::Type kDev, typename GridFunction, typename... Args>
void Launch(const std::string& operator_name, int device_id, void* stream,
            const ::infini::ops::Config* config_ptr,
            const Config& default_config,
            std::initializer_list<std::pair<std::string_view, std::uint64_t>>
                tuning_values,
            GridFunction grid_function, Args&&... args) {
  using CurrentBackend = detail::Backend<kDev>;

  auto device_guard = detail::ScopedDevice<kDev>::Create(device_id);
  assert(device_guard.has_value() && "Triton JIT failed to select a device.");
  if (!device_guard.has_value()) return;

  const detail::Target target = CurrentBackend::CurrentTarget();
  const std::string compilation_fingerprint =
      detail::CompilationFingerprint(operator_name);
  const bool context_valid =
      !target.backend.empty() && !target.architecture.empty() &&
      target.warp_size > 0 && !compilation_fingerprint.empty();
  assert(context_valid && "Triton JIT launch context is unavailable.");
  if (!context_valid) return;

  detail::ArgumentPack arguments;
  (arguments.Push(std::forward<Args>(args)), ...);
  assert(arguments.valid() && "Triton JIT arguments are invalid.");
  if (!arguments.valid()) return;

  const auto* auto_tuning_config_ptr =
      dynamic_cast<const detail::AutoTuningConfig*>(config_ptr);
  if (auto_tuning_config_ptr != nullptr) {
    detail::AutoTuningOptions options = auto_tuning_config_ptr->options();
    for (auto& candidate : options.candidates) {
      candidate = candidate.WithDefaultConstexprs(default_config);
    }

    const bool options_valid = !options.candidates.empty() &&
                               options.warmup_milliseconds >= 0 &&
                               options.repetition_milliseconds > 0;
    assert(options_valid && "Triton JIT auto-tuning options are invalid.");
    if (!options_valid) return;

    std::vector<std::string> key_names;
    std::vector<std::uint64_t> key_values;
    if (options.keys.empty()) {
      key_names.reserve(tuning_values.size());
      key_values.reserve(tuning_values.size());
      for (const auto& [name, value] : tuning_values) {
        key_names.emplace_back(name);
        key_values.push_back(value);
      }
    } else {
      key_names.reserve(options.keys.size());
      key_values.reserve(options.keys.size());
      for (const auto& requested_name : options.keys) {
        bool requested_twice = false;
        for (const auto& name : key_names) {
          if (name == requested_name) requested_twice = true;
        }

        std::size_t matches = 0;
        std::uint64_t matched_value = 0;
        for (const auto& [name, value] : tuning_values) {
          if (name == requested_name) {
            ++matches;
            matched_value = value;
          }
        }

        const bool key_valid = !requested_twice && matches == 1;
        assert(key_valid &&
               "Triton JIT auto-tuning keys must identify exactly one "
               "tuning value.");
        if (!key_valid) return;

        key_names.push_back(requested_name);
        key_values.push_back(matched_value);
      }
    }

    detail::LaunchAutoTuned<kDev>(target, device_id, compilation_fingerprint,
                                  operator_name, stream, options, key_names,
                                  key_values, grid_function, arguments);
    return;
  }

  const auto* jit_config_ptr = dynamic_cast<const Config*>(config_ptr);
  const Config& config =
      jit_config_ptr == nullptr ? default_config : *jit_config_ptr;
  const Config effective_config = config.WithDefaultConstexprs(default_config);
  detail::LaunchConfigured<kDev>(target, device_id, compilation_fingerprint,
                                 operator_name, stream, effective_config,
                                 grid_function, arguments);
}

}  // namespace infini::ops::triton::jit

#endif
