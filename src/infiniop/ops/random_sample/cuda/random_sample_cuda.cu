#include "../../../devices/cuda/cuda_handle.cuh"
#include "../info.h"
#include "random_sample_cuda.cuh"
#include "random_sample_kernel.cuh"

namespace op::random_sample::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t result_desc,
    infiniopTensorDescriptor_t probs_desc) {
    auto handle = reinterpret_cast<device::cuda::Handle *>(handle_);

    auto result = RandomSampleInfo::create(result_desc, probs_desc);
    CHECK_RESULT(result);

    auto info = result.take();
    size_t workspace_size;

#define CASE_P(CASE, Tidx, Tval)                                        \
    case CASE: {                                                        \
        auto workspace_result = calculateWorkspace<Tidx, Tval>(info.n); \
        CHECK_RESULT(workspace_result);                                 \
        workspace_size = workspace_result.take();                       \
    } break

#define CASE_I(CASE, Tidx)                          \
    case CASE:                                      \
        switch (info.dt_p) {                        \
            CASE_P(INFINI_DTYPE_F16, Tidx, half);   \
            CASE_P(INFINI_DTYPE_F32, Tidx, float);  \
            CASE_P(INFINI_DTYPE_F64, Tidx, double); \
        default:                                    \
            abort();                                \
        }                                           \
        break

    switch (info.dt_i) {
        CASE_I(INFINI_DTYPE_I8, int8_t);
        CASE_I(INFINI_DTYPE_I16, int16_t);
        CASE_I(INFINI_DTYPE_I32, int32_t);
        CASE_I(INFINI_DTYPE_I64, int64_t);
        CASE_I(INFINI_DTYPE_U8, uint8_t);
        CASE_I(INFINI_DTYPE_U16, uint16_t);
        CASE_I(INFINI_DTYPE_U32, uint32_t);
        CASE_I(INFINI_DTYPE_U64, uint64_t);
    default:
        abort();
    }

#undef CASE_I
#undef CASE_P

    *desc_ptr = new Descriptor(
        info,
        workspace_size,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::minWorkspaceSize() const {
    return _min_workspace_size;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream) const {

    if (workspace_size < _min_workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto block_size = _opaque->internal->blockSizeX();

    Calculate::calculate<Algo>(
        Algo{block_size}, _info, workspace, workspace_size,
        result, probs,
        random_val, topp, topk, temperature,
        stream);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::random_sample::cuda
