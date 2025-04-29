#include "swiglu_kunlun.h"

// Op interface declare
LAUNCH_ELEMENTWISE_KERNEL(SwiGLU)

namespace op::swiglu::kunlun {

typedef struct SwiGLUOp {
    static constexpr size_t num_inputs = 2;
    template <typename Tdata, typename... Args>
    static infiniStatus_t launch(Args... args) {
        launchSwiGLUKernel<Tdata>(args...);
        return INFINI_STATUS_SUCCESS;
    }
} SwiGLUOp;

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::kunlun::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &up_desc = input_desc_vec.at(0);
    const auto &gate_desc = input_desc_vec.at(1);
    const auto &out_shape = out_desc->shape();
    const auto &up_shape = up_desc->shape();
    const auto &gate_shape = gate_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F32);
    CHECK_SAME_SHAPE(out_shape, up_shape, gate_shape);

    // create KUNLUN elementwise descriptor
    CREATE_ELEMENTWISE_KUNLUN_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F32:
        return _device_info->calculate<SwiGLUOp, float>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu::kunlun
