#ifndef KV_CACHING_H
#define KV_CACHING_H

#include "../../../handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

#include "../../../../../build/ninetoothed/kv_caching.h"
#include "../../../ninetoothed/utils.h"

namespace op::kv_caching::ninetoothed {
class Descriptor final : public InfiniopDescriptor {

public:
    Descriptor(
        infiniopHandle_t handle,
        infiniopTensorDescriptor_t k_cache_desc,
        infiniopTensorDescriptor_t v_cache_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t past_kv_lengths_desc) : InfiniopDescriptor{handle->device, handle->device_id},
                                                           k_cache_shape_{k_cache_desc->shape()},
                                                           k_cache_strides_{k_cache_desc->strides()},
                                                           v_cache_shape_{v_cache_desc->shape()},
                                                           v_cache_strides_{v_cache_desc->strides()},
                                                           k_shape_{k_desc->shape()},
                                                           k_strides_{k_desc->strides()},
                                                           v_shape_{v_desc->shape()},
                                                           v_strides_{v_desc->strides()},
                                                           past_kv_lengths_shape_{past_kv_lengths_desc->shape()},
                                                           past_kv_lengths_strides_{past_kv_lengths_desc->strides()},
                                                           dtype_{k_desc->dtype()} {}

    ~Descriptor() = default;

    size_t get_workspace_size() const { return 0; };

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t k_cache,
        infiniopTensorDescriptor_t v_cache,
        infiniopTensorDescriptor_t k,
        infiniopTensorDescriptor_t v,
        infiniopTensorDescriptor_t past_kv_lengths) {
        *desc_ptr = new Descriptor{handle, k_cache, v_cache, k, v, past_kv_lengths};
        return INFINI_STATUS_SUCCESS;
    }

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *k_cache,
        void *v_cache,
        const void *k,
        const void *v,
        const void *past_kv_lengths,
        void *stream) const {
        auto k_cache_nt{::ninetoothed::Tensor{k_cache, k_cache_shape_, k_cache_strides_}};
        auto v_cache_nt{::ninetoothed::Tensor{v_cache, v_cache_shape_, v_cache_strides_}};
        auto k_nt{::ninetoothed::Tensor{k, k_shape_, k_strides_}};
        auto v_nt{::ninetoothed::Tensor{v, v_shape_, v_strides_}};
        auto past_kv_lengths_nt{::ninetoothed::Tensor{past_kv_lengths, past_kv_lengths_shape_, past_kv_lengths_strides_}};

        if (launch_kv_caching(stream,
                              k_cache_nt,
                              v_cache_nt,
                              k_nt,
                              v_nt,
                              past_kv_lengths_nt,
                              k_shape_[3],
                              dtype_,
                              64, 64)) {
            return INFINI_STATUS_NOT_IMPLEMENTED;
        }

        return INFINI_STATUS_SUCCESS;
    }

private:
    using Size = ::ninetoothed::Tensor<>::Size;
    using Stride = ::ninetoothed::Tensor<>::Stride;

    std::vector<Size> k_cache_shape_;
    std::vector<Stride> k_cache_strides_;

    std::vector<Size> v_cache_shape_;
    std::vector<Stride> v_cache_strides_;

    std::vector<Size> k_shape_;
    std::vector<Stride> k_strides_;
    std::vector<Size> v_shape_;
    std::vector<Stride> v_strides_;

    std::vector<Size> past_kv_lengths_shape_;
    std::vector<Stride> past_kv_lengths_strides_;

    infiniDtype_t dtype_;
};
} // namespace op::kv_caching::ninetoothed

#endif // KV_CACHING_H
