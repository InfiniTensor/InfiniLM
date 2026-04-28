#ifndef __REARRANGE_KUNLUN_H__
#define __REARRANGE_KUNLUN_H__

#include "../../../tensor.h"
#include "../rearrange.h"
#include <numeric>

namespace op::rearrange::kunlun {

struct RearrangeInfo {
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> src_strides;
    std::vector<ptrdiff_t> dst_strides;
    infiniDtype_t dtype;

    // Device space Size for shape, src_strides, dst_strides
    size_t workspace_size;

    size_t nelements() const {
        return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    }
    size_t ndim() const { return shape.size(); }
    size_t workspaceSize() const { return workspace_size; }

    static utils::Result<RearrangeInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc) {
        auto dtype = y_desc->dtype();
        auto ndim = y_desc->ndim();

        CHECK_OR_RETURN(x_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_OR_RETURN(x_desc->ndim() == ndim, INFINI_STATUS_BAD_TENSOR_SHAPE);

        auto y_shape = y_desc->shape();
        auto y_strides = y_desc->strides();
        auto x_shape = x_desc->shape();
        auto x_strides = x_desc->strides();

        CHECK_SAME_SHAPE(x_shape, y_shape);
        auto workspace_size_ = sizeof(size_t) * ndim + sizeof(ptrdiff_t) * ndim * 2;

        return utils::Result<RearrangeInfo>(RearrangeInfo{
            y_shape,
            x_strides,
            y_strides,
            dtype,
            workspace_size_,
        });
    }
};

class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;

    RearrangeInfo _info;

    Descriptor(
        Opaque *opaque,
        infiniDevice_t device_type,
        int device_id,
        RearrangeInfo info)
        : InfiniopDescriptor{device_type, device_id},
          _opaque(opaque),
          _info(info) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc);

    infiniStatus_t calculate(
        void *y,
        const void *x,
        void *stream) const;
};
} // namespace op::rearrange::kunlun

#endif // __REARRANGE_KUNLUN_H__
