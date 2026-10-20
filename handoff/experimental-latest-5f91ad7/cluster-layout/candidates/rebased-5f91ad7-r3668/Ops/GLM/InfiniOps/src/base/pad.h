#ifndef INFINI_OPS_BASE_PAD_H_
#define INFINI_OPS_BASE_PAD_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "operator.h"

namespace infini::ops {

// Preserve the ATen pad.out public signature, including its optional fill.
// Individual providers decide which padding modes and layouts they support.
class Pad : public Operator<Pad> {
 public:
  static constexpr bool kAutomaticDefault = true;

  Pad(const Tensor input, const std::vector<int64_t> pad,
      const std::string mode, const std::optional<double> value, Tensor out)
      : input_meta_(input),
        out_meta_(out),
        device_(input.device()),
        padding_(pad),
        mode_(mode),
        value_(value.value_or(0.0)) {}

  virtual void operator()(const Tensor input, const std::vector<int64_t> pad,
                          const std::string mode,
                          const std::optional<double> value,
                          Tensor out) const = 0;

 protected:
  const Tensor input_meta_, out_meta_;
  Device device_;
  std::vector<std::int64_t> padding_;
  std::string mode_;
  double value_;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_BASE_PAD_H_
