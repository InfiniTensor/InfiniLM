#ifndef INFINI_OPS_CAMBRICON_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_
#define INFINI_OPS_CAMBRICON_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "base/top_k_top_p_sampling_from_logits.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

template <typename T, typename OutputIndex>
void TopKTopPSamplingFromLogitsUnion(
    cnrtQueue_t queue, int core_per_cluster, int cluster_count, const T* logits,
    const int32_t* top_k, const float* top_p, const int32_t* row_indices,
    bool has_indices, bool joint, OutputIndex* out, int32_t batch_size,
    int32_t vocab_size, uint32_t seed, uint32_t offset);

template <>
class Operator<TopKTopPSamplingFromLogits, Device::Type::kCambricon>
    : public TopKTopPSamplingFromLogits {
 public:
  Operator(const Tensor logits, const Tensor top_k, const Tensor top_p,
           const std::optional<Tensor> indices,
           const std::string filter_apply_order, const bool deterministic,
           const bool check_nan, const std::optional<int64_t> seed,
           const std::optional<int64_t> offset, Tensor out)
      : TopKTopPSamplingFromLogits(logits, top_k, top_p, indices,
                                   filter_apply_order, deterministic, check_nan,
                                   seed, offset, out),
        logits_batch_size_{logits.size(0)},
        top_k_device_{
            cnrt_utils::AllocateDeviceBuffer(batch_size_ * sizeof(int32_t))},
        top_p_device_{
            cnrt_utils::AllocateDeviceBuffer(batch_size_ * sizeof(float))},
        row_indices_device_{
            cnrt_utils::AllocateDeviceBuffer(batch_size_ * sizeof(int32_t))} {
    ValidateSupportedOptions(logits, top_k, top_p, indices, filter_apply_order,
                             deterministic, check_nan, out);
    cnrt_utils::GetLaunchConfig(logits.device(), &core_per_cluster_,
                                &cluster_count_);
  }

  void operator()(const Tensor logits, const Tensor top_k, const Tensor top_p,
                  const std::optional<Tensor> indices,
                  const std::string filter_apply_order,
                  const bool deterministic, const bool check_nan,
                  const std::optional<int64_t> seed,
                  const std::optional<int64_t> offset,
                  Tensor out) const override {
    ValidateSupportedOptions(logits, top_k, top_p, indices, filter_apply_order,
                             deterministic, check_nan, out);
    if (batch_size_ == 0) {
      return;
    }
    PrepareParameters(top_k, top_p, indices);

    const uint64_t actual_seed = static_cast<uint64_t>(
        seed.value_or(static_cast<int64_t>(std::random_device{}())));
    const uint64_t actual_offset = static_cast<uint64_t>(offset.value_or(0));
    const uint32_t folded_seed = static_cast<uint32_t>(actual_seed) ^
                                 static_cast<uint32_t>(actual_seed >> 32);
    const uint32_t folded_offset = static_cast<uint32_t>(actual_offset) ^
                                   static_cast<uint32_t>(actual_offset >> 32);
    const auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);

    using LogitTypes =
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>;
    using OutputTypes = List<DataType::kInt32, DataType::kInt64>;
    DispatchFunc<Device::Type::kCambricon, LogitTypes, OutputTypes>(
        {logits.dtype(), out.dtype()},
        [&](auto logit_tag, auto output_tag) {
          using T = typename decltype(logit_tag)::type;
          using OutputIndex = typename decltype(output_tag)::type;
          TopKTopPSamplingFromLogitsUnion<T, OutputIndex>(
              queue, core_per_cluster_, cluster_count_,
              static_cast<const T*>(logits.data()),
              static_cast<const int32_t*>(top_k_device_.get()),
              static_cast<const float*>(top_p_device_.get()),
              static_cast<const int32_t*>(row_indices_device_.get()),
              indices.has_value(), filter_apply_order == "joint",
              static_cast<OutputIndex*>(out.data()),
              static_cast<int32_t>(batch_size_),
              static_cast<int32_t>(vocab_size_), folded_seed, folded_offset);
        },
        "CambriconTopKTopPSamplingFromLogits::operator()");
  }

 private:
  static void ValidateSupportedOptions(const Tensor logits, const Tensor top_k,
                                       const Tensor top_p,
                                       const std::optional<Tensor> indices,
                                       const std::string& filter_apply_order,
                                       const bool deterministic,
                                       const bool check_nan, const Tensor out) {
    assert((logits.dtype() == DataType::kFloat16 ||
            logits.dtype() == DataType::kBFloat16 ||
            logits.dtype() == DataType::kFloat32) &&
           "The Cambricon `TopKTopPSamplingFromLogits` provider requires "
           "float16, bfloat16, or float32 logits.");
    assert(logits.IsContiguous() && out.IsContiguous() &&
           "The Cambricon `TopKTopPSamplingFromLogits` provider requires "
           "contiguous logits and output.");
    assert(logits.device().type() == Device::Type::kCambricon &&
           out.device().type() == Device::Type::kCambricon &&
           logits.device().index() == out.device().index() &&
           "The Cambricon `TopKTopPSamplingFromLogits` provider requires "
           "logits and output on the same Cambricon device.");
    assert(top_k.device().type() == Device::Type::kCpu &&
           top_p.device().type() == Device::Type::kCpu &&
           top_k.IsContiguous() && top_p.IsContiguous() &&
           "The Cambricon `TopKTopPSamplingFromLogits` provider requires "
           "contiguous host-side top_k and top_p tensors.");
    assert((filter_apply_order == "top_k_first" ||
            filter_apply_order == "joint") &&
           "The Cambricon `TopKTopPSamplingFromLogits` provider supports "
           "top_k_first and joint filtering.");
    assert(deterministic &&
           "The Cambricon `TopKTopPSamplingFromLogits` provider supports "
           "only deterministic sampling.");
    assert(!check_nan &&
           "The Cambricon `TopKTopPSamplingFromLogits` provider does not "
           "support check_nan.");
    if (indices) {
      assert(indices->IsContiguous() &&
             (indices->device().type() == Device::Type::kCpu ||
              (indices->device().type() == Device::Type::kCambricon &&
               indices->device().index() == logits.device().index())) &&
             "The Cambricon `TopKTopPSamplingFromLogits` provider requires "
             "contiguous host or same-device indices.");
    }
  }

  void PrepareParameters(const Tensor top_k, const Tensor top_p,
                         const std::optional<Tensor> indices) const {
    std::vector<int32_t> top_k_host(batch_size_);
    std::vector<float> top_p_host(batch_size_);
    for (Tensor::Size row = 0; row < batch_size_; ++row) {
      const int64_t k = ReadHostInteger(top_k, row);
      top_k_host[row] = k <= 0 || static_cast<Tensor::Size>(k) > vocab_size_
                            ? static_cast<int32_t>(vocab_size_)
                            : static_cast<int32_t>(k);

      const double p = ReadHostFloat(top_p, row);
      top_p_host[row] =
          p <= 0.0 || p > 1.0
              ? 1.0F
              : static_cast<float>(std::max(
                    p, static_cast<double>(std::numeric_limits<float>::min())));
    }

    CNRT_CHECK(cnrtMemcpy(top_k_device_.get(), top_k_host.data(),
                          top_k_host.size() * sizeof(int32_t),
                          cnrtMemcpyHostToDev));
    CNRT_CHECK(cnrtMemcpy(top_p_device_.get(), top_p_host.data(),
                          top_p_host.size() * sizeof(float),
                          cnrtMemcpyHostToDev));

    if (!indices) {
      return;
    }

    std::vector<int64_t> indices_host(batch_size_);
    if (indices->device().type() == Device::Type::kCpu) {
      for (Tensor::Size row = 0; row < batch_size_; ++row) {
        indices_host[row] = ReadHostInteger(*indices, row);
      }
    } else if (indices->dtype() == DataType::kInt32) {
      std::vector<int32_t> source(batch_size_);
      CNRT_CHECK(cnrtMemcpy(source.data(), const_cast<void*>(indices->data()),
                            source.size() * sizeof(int32_t),
                            cnrtMemcpyDevToHost));
      std::copy(source.begin(), source.end(), indices_host.begin());
    } else {
      CNRT_CHECK(cnrtMemcpy(
          indices_host.data(), const_cast<void*>(indices->data()),
          indices_host.size() * sizeof(int64_t), cnrtMemcpyDevToHost));
    }

    std::vector<int32_t> normalized_indices(batch_size_);
    for (Tensor::Size row = 0; row < batch_size_; ++row) {
      assert(indices_host[row] >= 0 &&
             static_cast<Tensor::Size>(indices_host[row]) <
                 logits_batch_size_ &&
             "The Cambricon `TopKTopPSamplingFromLogits` provider received "
             "an out-of-range index.");
      normalized_indices[row] = static_cast<int32_t>(indices_host[row]);
    }
    CNRT_CHECK(cnrtMemcpy(row_indices_device_.get(), normalized_indices.data(),
                          normalized_indices.size() * sizeof(int32_t),
                          cnrtMemcpyHostToDev));
  }

  static int64_t ReadHostInteger(const Tensor tensor, Tensor::Size row) {
    const auto offset = row * tensor.stride(0);
    return tensor.dtype() == DataType::kInt32
               ? static_cast<const int32_t*>(tensor.data())[offset]
               : static_cast<const int64_t*>(tensor.data())[offset];
  }

  static double ReadHostFloat(const Tensor tensor, Tensor::Size row) {
    const auto offset = row * tensor.stride(0);
    switch (tensor.dtype()) {
      case DataType::kFloat16:
        return static_cast<const Float16*>(tensor.data())[offset].ToFloat();
      case DataType::kBFloat16:
        return static_cast<const BFloat16*>(tensor.data())[offset].ToFloat();
      case DataType::kFloat32:
        return static_cast<const float*>(tensor.data())[offset];
      case DataType::kFloat64:
        return static_cast<const double*>(tensor.data())[offset];
      default:
        assert(false &&
               "`TopKTopPSamplingFromLogits` received unsupported top_p "
               "dtype.");
        return 1.0;
    }
  }

  Tensor::Size logits_batch_size_{0};
  cnrt_utils::DeviceBuffer top_k_device_;
  cnrt_utils::DeviceBuffer top_p_device_;
  cnrt_utils::DeviceBuffer row_indices_device_;
  int core_per_cluster_{0};
  int cluster_count_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CAMBRICON_TOP_K_TOP_P_SAMPLING_FROM_LOGITS_KERNEL_H_
