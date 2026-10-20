#ifndef INFINI_OPS_LINKED_TORCH_OPS_MOE_WNA16_GEMM_H_
#define INFINI_OPS_LINKED_TORCH_OPS_MOE_WNA16_GEMM_H_

#include <optional>
#include <utility>

#include "base/moe_wna16_gemm.h"
#include "torch/tensor_.h"

namespace infini::ops::linked::torch {

template <typename Backend>
class TorchMoeWna16Gemm : public ::infini::ops::MoeWna16Gemm {
 public:
  TorchMoeWna16Gemm(const Tensor input, const Tensor b_qweight,
                    const Tensor b_scales, std::optional<Tensor> b_qzeros,
                    std::optional<Tensor> topk_weights,
                    const Tensor sorted_token_ids, const Tensor expert_ids,
                    const Tensor num_tokens_post_pad, const int64_t top_k,
                    const int64_t block_size_m, const int64_t block_size_n,
                    const int64_t block_size_k, const int64_t bit,
                    Tensor output)
      : ::infini::ops::MoeWna16Gemm{input,        b_qweight,
                                    b_scales,     b_qzeros,
                                    topk_weights, sorted_token_ids,
                                    expert_ids,   num_tokens_post_pad,
                                    top_k,        block_size_m,
                                    block_size_n, block_size_k,
                                    bit,          output},
        input_metadata_{input},
        b_qweight_metadata_{b_qweight},
        b_scales_metadata_{b_scales},
        b_qzeros_metadata_{b_qzeros},
        topk_weights_metadata_{topk_weights},
        sorted_token_ids_metadata_{sorted_token_ids},
        expert_ids_metadata_{expert_ids},
        num_tokens_post_pad_metadata_{num_tokens_post_pad},
        output_metadata_{output} {}

  using ::infini::ops::MoeWna16Gemm::operator();

  void operator()(const Tensor input, const Tensor b_qweight,
                  const Tensor b_scales, std::optional<Tensor> b_qzeros,
                  std::optional<Tensor> topk_weights,
                  const Tensor sorted_token_ids, const Tensor expert_ids,
                  const Tensor num_tokens_post_pad, const int64_t top_k,
                  const int64_t block_size_m, const int64_t block_size_n,
                  const int64_t block_size_k, const int64_t bit,
                  Tensor output) const override {
    ValidateCallMetadata(input, b_qweight, b_scales, b_qzeros, topk_weights,
                         sorted_token_ids, expert_ids, num_tokens_post_pad,
                         top_k, block_size_m, block_size_n, block_size_k, bit,
                         output);

    const typename Backend::StreamGuard stream_guard{
        Backend::GetStreamFromExternal(stream_, device_index_)};
    auto at_input = ToAten(input, input_metadata_);
    auto at_b_qweight = ToAten(b_qweight, b_qweight_metadata_);
    auto at_b_scales = ToAten(b_scales, b_scales_metadata_);
    auto at_b_qzeros = ToOptionalAten(b_qzeros, b_qzeros_metadata_);
    auto at_topk_weights = ToOptionalAten(topk_weights, topk_weights_metadata_);
    auto at_sorted_token_ids =
        ToAten(sorted_token_ids, sorted_token_ids_metadata_);
    auto at_expert_ids = ToAten(expert_ids, expert_ids_metadata_);
    auto at_num_tokens_post_pad =
        ToAten(num_tokens_post_pad, num_tokens_post_pad_metadata_);
    auto at_output = ToAten(output, output_metadata_);

    Backend::Call(std::move(at_input), std::move(at_output),
                  std::move(at_b_qweight), std::move(at_b_scales),
                  std::move(at_b_qzeros), std::move(at_topk_weights),
                  std::move(at_sorted_token_ids), std::move(at_expert_ids),
                  std::move(at_num_tokens_post_pad), top_k, block_size_m,
                  block_size_n, block_size_k, bit);
  }

 private:
  at::Tensor ToAten(const Tensor tensor, const Tensor metadata) const {
    return ToAtenTensor<Backend::kDeviceType>(
        const_cast<void*>(tensor.data()), metadata.shape(), metadata.strides(),
        metadata.dtype(), device_index_);
  }

  std::optional<at::Tensor> ToOptionalAten(
      const std::optional<Tensor>& tensor,
      const std::optional<Tensor>& metadata) const {
    if (!tensor) {
      return std::nullopt;
    }

    return ToAten(*tensor, *metadata);
  }

  Tensor input_metadata_;

  Tensor b_qweight_metadata_;

  Tensor b_scales_metadata_;

  std::optional<Tensor> b_qzeros_metadata_;

  std::optional<Tensor> topk_weights_metadata_;

  Tensor sorted_token_ids_metadata_;

  Tensor expert_ids_metadata_;

  Tensor num_tokens_post_pad_metadata_;

  Tensor output_metadata_;
};

}  // namespace infini::ops::linked::torch

#endif  // INFINI_OPS_LINKED_TORCH_OPS_MOE_WNA16_GEMM_H_
