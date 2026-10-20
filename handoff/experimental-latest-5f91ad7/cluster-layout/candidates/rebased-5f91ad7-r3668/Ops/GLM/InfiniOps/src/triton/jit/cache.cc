#include "triton/jit/cache.h"

#include <pybind11/pybind11.h>

#include <cassert>
#include <cstdio>
#include <exception>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace py = pybind11;

namespace infini::ops::triton::jit::detail {
namespace {

class IdentityBuilder {
 public:
  IdentityBuilder& Append(std::string_view value) {
    identity_ += std::to_string(value.size());
    identity_.push_back(':');
    identity_.append(value.data(), value.size());
    identity_.push_back('|');
    return *this;
  }

  template <typename T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
  IdentityBuilder& Append(T value) {
    return Append(std::to_string(value));
  }

  IdentityBuilder& Append(const Config& config) {
    Append(config.num_warps());
    Append(config.num_stages());
    Append(config.constexprs().size());
    for (const auto& [name, value] : config.constexprs()) {
      Append(name);
      Append(value);
    }
    return *this;
  }

  const std::string& identity() const { return identity_; }

  std::string Take() { return std::move(identity_); }

 private:
  std::string identity_;
};

py::object CacheFunction(const char* name) {
  return py::module_::import("infini.triton.jit.compile").attr(name);
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

Config ParseConfig(const py::dict& config) {
  Config::Constexprs constexprs;
  for (const auto& item : config["constexprs"].cast<py::dict>()) {
    constexprs.emplace(item.first.cast<std::string>(), item.second.cast<int>());
  }

  return {config["num_warps"].cast<unsigned>(),
          config["num_stages"].cast<unsigned>(), std::move(constexprs)};
}

template <typename Result, typename Function>
std::optional<Result> RunPython(const char* operation, Function&& function) {
  assert(Py_IsInitialized() != 0 &&
         "Triton JIT cache access requires an initialized Python interpreter.");
  if (Py_IsInitialized() == 0) return std::nullopt;

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

}  // namespace

KernelCacheKey KernelCacheKey::Build(const Target& target, int device_id,
                                     const std::string& compilation_fingerprint,
                                     const std::string& operator_name,
                                     const std::string& signature,
                                     const Config& config) {
  IdentityBuilder builder;
  builder.Append("kernel-v1")
      .Append(target.backend)
      .Append(target.architecture)
      .Append(target.warp_size)
      .Append(compilation_fingerprint)
      .Append(operator_name)
      .Append(signature)
      .Append(config);

  std::string identity = builder.identity();
  builder.Append(device_id);
  return {std::move(identity), builder.Take()};
}

std::string KernelCacheKey::ArtifactPrefix() const {
  auto prefix = RunPython<std::string>("cache path lookup", [&] {
    return CacheFunction("get_kernel_artifact_prefix")(identity_)
        .cast<std::string>();
  });
  assert(prefix.has_value() && "Triton JIT cache path lookup failed.");
  return prefix.value_or(std::string{});
}

AutoTuningCacheKey AutoTuningCacheKey::Build(
    const Target& target, const std::string& compilation_fingerprint,
    const std::string& operator_name, const std::string& signature,
    const std::vector<std::string>& key_names,
    const std::vector<std::uint64_t>& key_values,
    const std::vector<Config>& candidates, const std::vector<Grid>& grids,
    int warmup_milliseconds, int repetition_milliseconds) {
  IdentityBuilder builder;
  builder.Append("auto-tuning-v3")
      .Append(target.backend)
      .Append(target.architecture)
      .Append(target.warp_size)
      .Append(compilation_fingerprint)
      .Append(operator_name)
      .Append(signature)
      .Append(warmup_milliseconds)
      .Append(repetition_milliseconds)
      .Append(key_names.size());
  for (const auto& name : key_names) builder.Append(name);

  builder.Append(key_values.size());
  for (const auto value : key_values) builder.Append(value);

  builder.Append(candidates.size());
  for (const auto& candidate : candidates) builder.Append(candidate);

  builder.Append(grids.size());
  for (const auto grid : grids) {
    builder.Append(grid.x).Append(grid.y).Append(grid.z);
  }
  return AutoTuningCacheKey{builder.Take()};
}

std::optional<KernelArtifact> ReadKernelArtifact(
    const std::string& output_prefix, const std::string& expected_identity) {
  auto artifact = RunPython<std::optional<KernelArtifact>>(
      "artifact lookup", [&]() -> std::optional<KernelArtifact> {
        const py::object result = CacheFunction("read_kernel_artifact")(
            output_prefix, expected_identity);
        if (result.is_none()) return std::nullopt;

        const py::dict fields = result.cast<py::dict>();
        KernelMetadata metadata{
            fields["name"].cast<std::string>(),
            fields["num_warps"].cast<unsigned>(),
            fields["shared_memory_size"].cast<unsigned>(),
            fields["global_scratch_size"].cast<int>(),
            fields["profile_scratch_size"].cast<int>(),
        };
        return KernelArtifact{std::move(metadata),
                              fields["binary"].cast<std::string>()};
      });
  assert(artifact.has_value() && "Triton JIT artifact lookup failed.");
  return artifact.value_or(std::nullopt);
}

AutoTuningCache& AutoTuningCache::Instance() {
  static AutoTuningCache cache;
  return cache;
}

std::optional<Config> AutoTuningCache::Find(const AutoTuningCacheKey& key) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = entries_.find(key.identity());
    if (it != entries_.end()) return it->second;
  }

  auto stored = RunPython<std::optional<Config>>(
      "auto-tuning cache lookup", [&]() -> std::optional<Config> {
        const py::object result =
            CacheFunction("read_auto_tuning_config")(key.identity());
        if (result.is_none()) return std::nullopt;
        return ParseConfig(result.cast<py::dict>());
      });
  assert(stored.has_value() && "Triton JIT auto-tuning cache lookup failed.");
  if (!stored.has_value() || !stored->has_value()) return std::nullopt;

  const std::lock_guard<std::mutex> lock(mutex_);
  return entries_.try_emplace(key.identity(), std::move(stored->value()))
      .first->second;
}

void AutoTuningCache::Insert(const AutoTuningCacheKey& key,
                             const Config& config) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    entries_.insert_or_assign(key.identity(), config);
  }

  const auto written = RunPython<bool>("auto-tuning cache write", [&] {
    CacheFunction("write_auto_tuning_config")(key.identity(),
                                              BuildConfig(config));
    return true;
  });
  assert(written.has_value() && "Triton JIT auto-tuning cache write failed.");
}

}  // namespace infini::ops::triton::jit::detail
