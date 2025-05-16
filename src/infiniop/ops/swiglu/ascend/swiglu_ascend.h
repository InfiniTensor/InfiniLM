#ifndef __ACLNN_SWIGLU_H__
#define __ACLNN_SWIGLU_H__

#include "../../../../utils.h"
#include "../../../../utils/check.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::swiglu::ascend {
class SwigluInfo {

    SwigluInfo() = default;

public:
    infiniDtype_t dtype;
    std::vector<size_t> shape;
    int32_t ndim;
    std::vector<ptrdiff_t> c_strides;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;

    static utils::Result<SwigluInfo> create(infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t b_desc) {
        CHECK_OR_RETURN(c_desc && a_desc && b_desc, INFINI_STATUS_BAD_PARAM);
        CHECK_OR_RETURN(!c_desc->hasBroadcastDim(), INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(c_desc->ndim() == a_desc->ndim()
                            && c_desc->ndim() == b_desc->ndim()
                            && (c_desc->ndim() == 2 || c_desc->ndim() == 3),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_SAME_SHAPE(c_desc->shape(), a_desc->shape(), b_desc->shape());
        int32_t ndim = c_desc->ndim();
        CHECK_OR_RETURN(c_desc->stride(ndim - 1) == 1
                            && a_desc->stride(ndim - 1) == 1
                            && b_desc->stride(ndim - 1) == 1,
                        INFINI_STATUS_BAD_TENSOR_STRIDES);
        CHECK_OR_RETURN(c_desc->dtype() == a_desc->dtype()
                            && c_desc->dtype() == b_desc->dtype(),
                        INFINI_STATUS_BAD_TENSOR_DTYPE);

        return utils::Result<SwigluInfo>(SwigluInfo{
            c_desc->dtype(),
            c_desc->shape(),
            ndim,
            c_desc->strides(),
            a_desc->strides(),
            b_desc->strides(),
        });
    }
};

class Descriptor final : public InfiniopDescriptor {
    SwigluInfo _info;
    size_t _workspace_size;

    Descriptor(SwigluInfo info, size_t workspace_size, infiniDevice_t device_type, int device_id) : InfiniopDescriptor{device_type, device_id},
                                                                                                    _info(info), _workspace_size(workspace_size) {}

public:
    ~Descriptor();
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                 infiniopTensorDescriptor_t c_desc,
                                 std::vector<infiniopTensorDescriptor_t> input_descs);
    size_t workspaceSize() const { return _workspace_size; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        std::vector<const void *> inputs,
        void *stream) const;
};

extern "C" infiniStatus_t swiglu_kernel_launch(
    void *c, void *a, void *b,
    infiniDtype_t dtype, size_t batch, size_t seq, size_t hd,
    ptrdiff_t stride_batch_c, ptrdiff_t stride_batch_a, ptrdiff_t stride_batch_b,
    ptrdiff_t stride_seq_c, ptrdiff_t stride_seq_a, ptrdiff_t stride_seq_b, void *stream);

} // namespace op::swiglu::ascend
#endif // __ACLNN_SWIGLU_H__
