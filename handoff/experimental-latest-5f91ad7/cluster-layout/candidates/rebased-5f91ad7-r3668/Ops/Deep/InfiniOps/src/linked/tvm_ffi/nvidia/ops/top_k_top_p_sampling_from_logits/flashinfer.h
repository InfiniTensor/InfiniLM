#ifndef INFINI_OPS_LINKED_TVM_FFI_NVIDIA_OPS_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_FLASHINFER_H_
#define INFINI_OPS_LINKED_TVM_FFI_NVIDIA_OPS_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_FLASHINFER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

#include "base/top_k_top_p_sampling_from_logits.h"

namespace infini::ops {

template <>
class Operator<TopKTopPSamplingFromLogits, Device::Type::kNvidia, 16>
    : public TopKTopPSamplingFromLogits {
 public:
  Operator(const Tensor logits, const Tensor top_k, const Tensor top_p,
           const std::optional<Tensor> indices,
           const std::string filter_apply_order, const bool deterministic,
           const bool check_nan, const std::optional<int64_t> seed,
           const std::optional<int64_t> offset, Tensor out);

  ~Operator() override;

  std::size_t workspace_size_in_bytes() const override;

  void operator()(const Tensor logits, const Tensor top_k, const Tensor top_p,
                  const std::optional<Tensor> indices,
                  const std::string filter_apply_order,
                  const bool deterministic, const bool check_nan,
                  const std::optional<int64_t> seed,
                  const std::optional<int64_t> offset,
                  Tensor out) const override;

 private:
  struct StagingSlot {
    float* top_p{nullptr};
    int64_t* top_k{nullptr};
    void* indices{nullptr};
    void* event{nullptr};
    bool event_recorded{false};
  };

  std::size_t workspace_size_{0};

  Tensor::Size logits_batch_size_{0};

  int device_index_{0};

  DataType top_k_dtype_;

  DataType top_p_dtype_;

  DataType out_dtype_;

  std::optional<DataType> indices_dtype_;

  std::optional<Device> indices_device_;

  std::string filter_apply_order_;

  bool deterministic_{false};

  mutable std::array<StagingSlot, 2> staging_slots_;

  mutable std::size_t next_staging_slot_{0};

  mutable std::mutex mutex_;

  void ReleaseStagingSlots() noexcept;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TVM_FFI_NVIDIA_OPS_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_FLASHINFER_H_
