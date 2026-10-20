#ifndef INFINI_OPS_BASE_MAMBA_SELECTIVE_SCAN_H_
#define INFINI_OPS_BASE_MAMBA_SELECTIVE_SCAN_H_

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "operator.h"

namespace infini::ops {
// Ordering deliberately matches the original infiniop descriptor/call ABI.
// data() denotes the logical origin; no storage-base offset is added again.
struct MambaScanLayout {
  std::size_t batch, length, channels, states;
  std::ptrdiff_t strides[10][3]{};
  bool contiguous;
};

class MambaSelectiveScan : public Operator<MambaSelectiveScan> {
 public:
  MambaSelectiveScan(Tensor out, const Tensor x, const Tensor dt,
                     const Tensor b, const Tensor c, const Tensor a_log,
                     const Tensor d, const Tensor gate, const Tensor dt_bias,
                     Tensor state)
      : layout_{ValidateMetadata(
            {out, x, dt, b, c, a_log, d, gate, dt_bias, state})},
        dtype_{x.dtype()},
        device_{x.device()} {}
  virtual void operator()(Tensor out, const Tensor x, const Tensor dt,
                          const Tensor b, const Tensor c, const Tensor a_log,
                          const Tensor d, const Tensor gate,
                          const Tensor dt_bias, Tensor state) const = 0;

  static MambaScanLayout ValidateMetadata(const std::array<Tensor, 10>& ts) {
    const auto dtype = ts[1].dtype();
    if (dtype != DataType::kFloat32 && dtype != DataType::kFloat16 &&
        dtype != DataType::kBFloat16)
      throw std::invalid_argument(
          "MambaSelectiveScan: expected F32/F16/BF16 inputs");
    const auto parameter_dtype = ts[5].dtype();
    if (ts[6].dtype() != parameter_dtype ||
        (parameter_dtype != dtype && parameter_dtype != DataType::kFloat32))
      throw std::invalid_argument(
          "MambaSelectiveScan: A_log/D must share input dtype or F32");
    constexpr int ranks[10] = {3, 3, 3, 3, 3, 2, 1, 3, 1, 3};
    for (int i = 0; i < 10; ++i) {
      if (ts[i].ndim() != static_cast<std::size_t>(ranks[i]) ||
          ts[i].device() != ts[1].device() ||
          ts[i].dtype() != (i == 9               ? DataType::kFloat32
                            : (i == 5 || i == 6) ? parameter_dtype
                                                 : dtype))
        throw std::invalid_argument(
            "MambaSelectiveScan: incompatible rank/device/dtype");
      SpanBytes(ts[i]);  // Reject negative strides and overflow, including
                         // unused rows.
    }
    const auto batch = ts[1].size(0), length = ts[1].size(1),
               channels = ts[1].size(2), states = ts[3].size(2);
    for (int i : {0, 2, 7})
      if (ts[i].shape() != ts[1].shape())
        throw std::invalid_argument(
            "MambaSelectiveScan: x/dt/gate/out shape mismatch");
    if (ts[3].size(0) != batch || ts[3].size(1) != length ||
        ts[4].shape() != ts[3].shape() || ts[5].size(0) != channels ||
        ts[5].size(1) != states || ts[6].size(0) != channels ||
        ts[8].size(0) != channels || ts[9].size(0) < batch ||
        ts[9].size(1) != channels || ts[9].size(2) != states)
      throw std::invalid_argument(
          "MambaSelectiveScan: B/C/A_log/D/bias/state shape mismatch");
    // The recurrence may use a larger state pool; only the first batch rows
    // change.
    ValidateWritable(ts[0]);
    ValidateWritable(ts[9]);
    if (channels && batch > std::numeric_limits<std::size_t>::max() / channels)
      throw std::overflow_error("MambaSelectiveScan: launch size overflow");
    MambaScanLayout layout{batch, length, channels, states, {}, true};
    for (int i = 0; i < 10; ++i) {
      std::size_t expected = 1;
      for (int axis = ranks[i] - 1; axis >= 0; --axis) {
        layout.strides[i][axis] = ts[i].stride(axis);
        if (ts[i].size(axis) > 1 &&
            ts[i].stride(axis) != static_cast<std::ptrdiff_t>(expected))
          layout.contiguous = false;
        if (ts[i].size(axis) &&
            expected >
                std::numeric_limits<std::size_t>::max() / ts[i].size(axis))
          throw std::overflow_error(
              "MambaSelectiveScan: shape product overflow");
        expected *= ts[i].size(axis);
      }
    }
    return layout;
  }

 protected:
  MambaScanLayout layout_;
  DataType dtype_;
  Device device_;

 private:
  static std::size_t SpanBytes(const Tensor& t) {
    std::size_t last = 0;
    bool empty = false;
    const auto limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    for (std::size_t axis = 0; axis < t.ndim(); ++axis) {
      if (t.stride(axis) < 0)
        throw std::invalid_argument("MambaSelectiveScan: negative stride");
      if (!t.size(axis)) {
        empty = true;
        continue;
      }
      const auto stride = static_cast<std::size_t>(t.stride(axis));
      if (stride && t.size(axis) - 1 > (limit - last) / stride)
        throw std::overflow_error("MambaSelectiveScan: span overflow");
      last += (t.size(axis) - 1) * stride;
    }
    const auto width = t.dtype() == DataType::kFloat32 ? 4u : 2u;
    if (empty) return 0;
    if (last >= limit / width)
      throw std::overflow_error("MambaSelectiveScan: byte span overflow");
    return (last + 1) * width;
  }
  static void ValidateWritable(const Tensor& t) {
    // Monotone pitched rows with arbitrary positive inner stride. Broadcast or
    // overlapping writable dimensions are rejected before submitting a kernel.
    for (std::size_t axis = 0; axis < t.ndim(); ++axis)
      if (!t.size(axis)) return;
    std::size_t extent = 1;
    for (int axis = static_cast<int>(t.ndim()) - 1; axis >= 0; --axis) {
      if (t.size(axis) > 1 && static_cast<std::size_t>(t.stride(axis)) < extent)
        throw std::invalid_argument(
            "MambaSelectiveScan: overlapping writable layout");
      if (t.size(axis))
        extent += (t.size(axis) - 1) * static_cast<std::size_t>(t.stride(axis));
    }
  }
};
}  // namespace infini::ops
#endif
