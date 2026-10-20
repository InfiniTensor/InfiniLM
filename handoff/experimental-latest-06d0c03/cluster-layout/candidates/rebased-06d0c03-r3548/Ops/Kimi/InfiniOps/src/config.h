#ifndef INFINI_OPS_CONFIG_H_
#define INFINI_OPS_CONFIG_H_

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>

#include "cloneable.h"

namespace infini::ops {

class Config {
 public:
  virtual ~Config() = default;

  virtual std::unique_ptr<Config> Clone() const {
    return std::make_unique<Config>(*this);
  }

  // A typed configuration may require a provider that consumes its metadata
  // during construction. Legacy configurations and providers are unchanged.
  virtual bool requires_config_aware_provider() const { return false; }

  std::size_t implementation_index() const {
    return implementation_index_.value_or(0);
  }

  void set_implementation_index(std::size_t implementation_index) {
    implementation_index_ = implementation_index;
  }

  bool needs_implementation_resolution() const {
    return !implementation_index_.has_value();
  }

  // Opt-in metadata-based selection. Explicit indices retain their meaning.
  static constexpr std::size_t kAutomaticImplementation =
      std::numeric_limits<std::size_t>::max();

  static Config Automatic() {
    Config config;
    config.set_implementation_index(kAutomaticImplementation);
    return config;
  }

 private:
  std::optional<std::size_t> implementation_index_;
};

}  // namespace infini::ops

#endif
