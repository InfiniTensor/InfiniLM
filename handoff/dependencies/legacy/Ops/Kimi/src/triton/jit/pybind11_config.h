#ifndef INFINI_OPS_TRITON_JIT_PYBIND11_CONFIG_H_
#define INFINI_OPS_TRITON_JIT_PYBIND11_CONFIG_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <initializer_list>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "triton/jit/config_.h"

namespace infini::ops::triton::jit {
namespace detail {

inline bool IsKnownConfigField(
    const std::string& field,
    std::initializer_list<std::string_view> known_fields) {
  for (const auto known_field : known_fields) {
    if (field == known_field) return true;
  }

  return false;
}

inline void ValidateConfigFields(
    const pybind11::dict& config_dict,
    std::initializer_list<std::string_view> known_fields) {
  for (const auto& item : config_dict) {
    const auto field = item.first.cast<std::string>();
    if (!IsKnownConfigField(field, known_fields)) {
      throw pybind11::value_error("Unknown Triton JIT config field `" + field +
                                  "`.");
    }
  }
}

inline Config::Constexprs ParseConstexprs(
    const pybind11::dict& constexprs_dict) {
  Config::Constexprs constexprs;

  for (const auto& item : constexprs_dict) {
    constexprs.emplace(item.first.cast<std::string>(), item.second.cast<int>());
  }

  return constexprs;
}

inline Config ParseConfig(const pybind11::dict& config_dict) {
  ValidateConfigFields(config_dict, {"num_warps", "num_stages", "constexprs"});

  const Config defaults;
  const auto num_warps = config_dict.contains("num_warps")
                             ? config_dict["num_warps"].cast<unsigned>()
                             : defaults.num_warps();
  const auto num_stages = config_dict.contains("num_stages")
                              ? config_dict["num_stages"].cast<unsigned>()
                              : defaults.num_stages();
  auto constexprs =
      config_dict.contains("constexprs")
          ? ParseConstexprs(config_dict["constexprs"].cast<pybind11::dict>())
          : Config::Constexprs{};

  return Config{num_warps, num_stages, std::move(constexprs)};
}

inline AutoTuningOptions ParseAutoTuningOptions(
    const pybind11::dict& options_dict) {
  ValidateConfigFields(
      options_dict,
      {"warmup_milliseconds", "repetition_milliseconds", "keys", "candidates"});

  AutoTuningOptions options;

  if (options_dict.contains("warmup_milliseconds")) {
    options.warmup_milliseconds =
        options_dict["warmup_milliseconds"].cast<int>();
  }

  if (options_dict.contains("repetition_milliseconds")) {
    options.repetition_milliseconds =
        options_dict["repetition_milliseconds"].cast<int>();
  }

  if (options_dict.contains("keys")) {
    options.keys = options_dict["keys"].cast<std::vector<std::string>>();
  }

  if (options_dict.contains("candidates")) {
    for (const auto& candidate :
         options_dict["candidates"].cast<pybind11::list>()) {
      options.candidates.push_back(
          ParseConfig(candidate.cast<pybind11::dict>()));
    }
  }

  return options;
}

}  // namespace detail

inline std::unique_ptr<infini::ops::Config> ConfigFromPyDict(
    const pybind11::dict& config_dict) {
  if (!config_dict.contains("auto_tuning")) {
    return std::make_unique<Config>(detail::ParseConfig(config_dict));
  }

  detail::ValidateConfigFields(config_dict, {"auto_tuning"});
  auto options = detail::ParseAutoTuningOptions(
      config_dict["auto_tuning"].cast<pybind11::dict>());

  return std::make_unique<detail::AutoTuningConfig>(std::move(options));
}

}  // namespace infini::ops::triton::jit

#endif
