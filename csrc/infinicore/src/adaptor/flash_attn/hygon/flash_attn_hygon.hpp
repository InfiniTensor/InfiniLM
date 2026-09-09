#ifndef INFINICORE_ADAPTOR_FLASH_ATTN_HYGON_FLASH_ATTN_HYGON_HPP
#define INFINICORE_ADAPTOR_FLASH_ATTN_HYGON_FLASH_ATTN_HYGON_HPP

#if defined(ENABLE_FLASH_ATTN) && defined(ENABLE_HYGON_API)

#include <ATen/ATen.h>
#include <optional>
#include <vector>

namespace flash {

std::vector<at::Tensor>
mha_fwd_kvcache(at::Tensor &q,
                const at::Tensor &kcache,
                const at::Tensor &vcache,
                std::optional<const at::Tensor> &k_,
                std::optional<const at::Tensor> &v_,
                std::optional<const at::Tensor> &seqlens_k_,
                std::optional<const at::Tensor> &rotary_cos_,
                std::optional<const at::Tensor> &rotary_sin_,
                std::optional<const at::Tensor> &cache_batch_idx_,
                std::optional<const at::Tensor> &leftpad_k_,
                std::optional<at::Tensor> &block_table_,
                std::optional<at::Tensor> &alibi_slopes_,
                std::optional<at::Tensor> &out_,
                float softmax_scale,
                bool is_causal,
                int window_size_left,
                int window_size_right,
                float softcap,
                bool is_rotary_interleaved,
                int num_splits);

std::vector<at::Tensor>
vllm_mha_varlen_fwd(at::Tensor &q,
                    const at::Tensor &k,
                    const at::Tensor &v,
                    std::optional<at::Tensor> &out_,
                    const at::Tensor &cu_seqlens_q,
                    const at::Tensor &cu_seqlens_k,
                    std::optional<at::Tensor> &seqused_k,
                    std::optional<const at::Tensor> &leftpad_k_,
                    std::optional<at::Tensor> &block_table_,
                    std::optional<at::Tensor> &alibi_slopes_,
                    int max_seqlen_q,
                    int max_seqlen_k,
                    float p_dropout,
                    float softmax_scale,
                    bool zero_tensors,
                    bool is_causal,
                    int window_size_left,
                    int window_size_right,
                    float softcap,
                    bool return_softmax,
                    std::optional<at::Generator> gen_);

} // namespace flash

#endif // ENABLE_FLASH_ATTN && ENABLE_HYGON_API
#endif // INFINICORE_ADAPTOR_FLASH_ATTN_HYGON_FLASH_ATTN_HYGON_HPP
