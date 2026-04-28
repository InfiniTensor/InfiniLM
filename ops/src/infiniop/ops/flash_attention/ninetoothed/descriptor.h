#ifndef __FLASH_ATTENTION_DESCRIPTOR_H__
#define __FLASH_ATTENTION_DESCRIPTOR_H__

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/flash_attention.h"
#include "../../../ninetoothed/utils.h"

namespace op::flash_attention::ninetoothed {

class Descriptor final : public InfiniopDescriptor {
public:
    Descriptor(infiniopHandle_t handle,
               infiniopTensorDescriptor_t out_desc,
               infiniopTensorDescriptor_t q_desc,
               infiniopTensorDescriptor_t k_desc,
               infiniopTensorDescriptor_t v_desc,
               infiniopTensorDescriptor_t total_kv_len,
               double scale,
               char is_causal) : InfiniopDescriptor{handle->device, handle->device_id},
                                 _query_shape{q_desc->shape()},
                                 _query_strides{q_desc->strides()},
                                 _key_shape{k_desc->shape()},
                                 _key_strides{k_desc->strides()},
                                 _value_shape{v_desc->shape()},
                                 _value_strides{v_desc->strides()},
                                 _total_kv_shape{total_kv_len->shape()},
                                 _total_kv_strides{total_kv_len->strides()},
                                 _output_strides{out_desc->strides()},
                                 _dtype{q_desc->dtype()},
                                 _scale{scale},
                                 _is_causal{is_causal} {
    }

    ~Descriptor() = default;

    size_t get_workspace_size() const {
        return 0;
    }

    infiniStatus_t calculate(void *workspace,
                             size_t workspace_size,
                             void *out,
                             const void *q,
                             const void *k,
                             const void *v,
                             const void *total_kv_len,
                             void *stream) const {
        uint64_t empty_shape[4];
        int64_t empty_strides[4];

        auto query{::ninetoothed::Tensor{q, _query_shape, _query_strides}};
        auto key{::ninetoothed::Tensor{k, _key_shape, _key_strides}};
        auto value{::ninetoothed::Tensor{v, _value_shape, _value_strides}};
        auto total_kv_length{::ninetoothed::Tensor{total_kv_len, _total_kv_shape, _total_kv_strides}};

        NineToothedTensor attn_mask{nullptr, empty_shape, empty_strides};
        NineToothedTensor is_causal;
        NineToothedTensor scale{const_cast<double *>(&_scale), nullptr, nullptr};
        auto output{::ninetoothed::Tensor{out, _query_shape, _output_strides}};
        NineToothedTensor with_attn_mask;
        NineToothedTensor causal_variant;

        const auto with_kv_cache_{0};
        const auto emb_dim_{_query_shape[3]};
        const auto is_causal_{_is_causal};
        const auto with_attn_mask_{0};
        const auto causal_variant_{2};
        const auto dtype_{_dtype};

        constexpr auto block_size_m_{256};
        constexpr auto block_size_n_{64};

        if (launch_flash_attention(stream,
                                   query,
                                   key,
                                   value,
                                   total_kv_length,
                                   attn_mask,
                                   is_causal,
                                   scale,
                                   output,
                                   with_attn_mask,
                                   causal_variant,
                                   with_kv_cache_,
                                   emb_dim_,
                                   is_causal_,
                                   with_attn_mask_,
                                   causal_variant_,
                                   dtype_,
                                   block_size_m_,
                                   block_size_n_)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }

        return INFINI_STATUS_SUCCESS;
    }

    static infiniStatus_t create(infiniopHandle_t handle,
                                 Descriptor **desc,
                                 infiniopTensorDescriptor_t out_desc,
                                 infiniopTensorDescriptor_t q_desc,
                                 infiniopTensorDescriptor_t k_desc,
                                 infiniopTensorDescriptor_t v_desc,
                                 infiniopTensorDescriptor_t total_kv_len,
                                 double scale,
                                 char is_causal) {
        *desc = new Descriptor{handle, out_desc, q_desc, k_desc, v_desc, total_kv_len, scale, is_causal};

        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;

    using Stride = ::ninetoothed::Tensor<>::Stride;

    std::vector<Size> _query_shape;

    std::vector<Stride> _query_strides;

    std::vector<Size> _key_shape;

    std::vector<Stride> _key_strides;

    std::vector<Size> _value_shape;

    std::vector<Stride> _value_strides;

    std::vector<Size> _total_kv_shape;

    std::vector<Stride> _total_kv_strides;

    std::vector<Stride> _output_strides;

    infiniDtype_t _dtype;

    double _scale;

    char _is_causal;
};

} // namespace op::flash_attention::ninetoothed

#endif // __FLASH_ATTENTION_DESCRIPTOR_H__
