#include "torch/nvidia/c10.h"
#include "torch/ops/flash_attn_varlen_func/aten_impl.h"

namespace infini::ops {

namespace detail {

template <>
struct AtenFlashAttnVarlenPolicy<Device::Type::kNvidia> {
  static void ValidateWindow(const std::vector<int64_t>&) {}

  static AtenFlashAttnVarlenOptions MakeOptions(
      bool causal, const std::vector<int64_t>& window_size, const at::Tensor&,
      const at::Tensor&) {
    AtenFlashAttnVarlenOptions options;

    if (window_size[0] >= 0) {
      options.window_size_left = window_size[0];
    }

    if (causal) {
      options.window_size_right = 0;
    } else if (window_size[1] >= 0) {
      options.window_size_right = window_size[1];
    }

    return options;
  }
};

}  // namespace detail

template class AtenFlashAttnVarlenFunc<Device::Type::kNvidia>;

}  // namespace infini::ops
