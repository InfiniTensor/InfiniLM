#ifndef INFINI_OPS_TRITON_JIT_CONFIG_H_
#define INFINI_OPS_TRITON_JIT_CONFIG_H_

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config.h"

namespace infini::ops::triton::jit {

class Config : public Cloneable<infini::ops::Config, Config> {
 public:
  using Constexprs = std::map<std::string, int, std::less<>>;

  Config() = default;

  Config(unsigned num_warps, unsigned num_stages, Constexprs constexprs)
      : num_warps_(num_warps),
        num_stages_(num_stages),
        constexprs_(std::move(constexprs)) {}

  unsigned num_warps() const { return num_warps_; }

  unsigned num_stages() const { return num_stages_; }

  const Constexprs& constexprs() const { return constexprs_; }

  std::optional<int> FindConstexpr(std::string_view name) const {
    const auto it = constexprs_.find(name);
    return it == constexprs_.end() ? std::nullopt
                                   : std::optional<int>{it->second};
  }

  Config WithDefaultConstexprs(const Config& defaults) const {
    Config result = *this;
    for (const auto& [name, value] : defaults.constexprs_) {
      result.constexprs_.try_emplace(name, value);
    }
    return result;
  }

 private:
  unsigned num_warps_{4};

  unsigned num_stages_{3};

  Constexprs constexprs_;
};

namespace detail {

struct AutoTuningOptions {
  std::vector<std::string> keys;

  std::vector<Config> candidates;

  int warmup_milliseconds{25};

  int repetition_milliseconds{100};
};

class AutoTuningConfig : public Cloneable<Config, AutoTuningConfig> {
 public:
  explicit AutoTuningConfig(AutoTuningOptions options)
      : options_(std::move(options)) {}

  const AutoTuningOptions& options() const { return options_; }

 private:
  AutoTuningOptions options_;
};

}  // namespace detail

}  // namespace infini::ops::triton::jit

#endif
