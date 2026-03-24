#pragma once

#include "../../../backends/attention_backends.hpp"
#include "../../../engine/forward_context.hpp"
#include "infinicore/tensor.hpp"
#include "flash_attn.hpp"
#include "paged_attn.hpp"
#include "static_attn.hpp"
#include <memory>
#include <stdexcept>
#include <variant>

namespace infinilm::layers::attention {
/*
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/attention.py

class Attention(nn.Module, AttentionLayerBase):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()

        if attn_backend is None:
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                use_mla=False,
                has_sink=self.has_sink,
                use_mm_prefix=self.use_mm_prefix,
                use_per_head_quant_scales=use_per_head_quant_scales,
                attn_type=attn_type,
            )


        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **extra_impl_args,
        )


    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        pass
*/

/*
Attention layer.

This class takes query, key, and value tensors as input.
The input tensors can either contain prompt tokens or generation tokens.
The class does the following:

1. Store the input key and value tensors in the KV cache.
2. Perform (multi-head/multi-query/grouped-query) attention.
3. Return the output tensor.
*/

using AttentionImpl = std::variant<std::shared_ptr<backends::StaticAttentionImpl>, std::shared_ptr<backends::PagedAttentionImpl>, std::shared_ptr<backends::FlashAttentionImpl>>;

class AttentionLayer {
public:
    AttentionLayer(size_t num_heads,
                   size_t head_size,
                   float scale,
                   size_t num_kv_heads,
                   size_t layer_idx,
                   ::infinilm::backends::AttentionBackend attention_backend) {
        switch (attention_backend) {
        case ::infinilm::backends::AttentionBackend::STATIC_ATTN:
            attn_backend_impl_ = std::make_shared<backends::StaticAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
            break;
        case ::infinilm::backends::AttentionBackend::PAGED_ATTN:
            attn_backend_impl_ = std::make_shared<backends::PagedAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
            break;
        case ::infinilm::backends::AttentionBackend::FLASH_ATTN:
            attn_backend_impl_ = std::make_shared<backends::FlashAttentionImpl>(num_heads, head_size, scale, num_kv_heads, layer_idx);
            break;
        default:
            throw std::runtime_error("infinilm::layers::attention::AttentionLayer: unsupported attention backend");
        }
    }

    inline infinicore::Tensor forward(const infinicore::Tensor &query,
                                      const infinicore::Tensor &key,
                                      const infinicore::Tensor &value,
                                      std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                                      const infinilm::engine::AttentionMetadata &attn_metadata) const {
        return std::visit(
            [&](auto &impl_ptr) -> infinicore::Tensor {
                return impl_ptr->forward(nullptr, query, key, value, kv_cache, attn_metadata);
            },
            attn_backend_impl_);
    }

private:
    AttentionImpl attn_backend_impl_;
};
} // namespace infinilm::layers::attention
