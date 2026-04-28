#ifndef __INFINIOP_ARGWHERE_H__
#define __INFINIOP_ARGWHERE_H__
#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include "infinicore.h"
#include <cstddef>

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::argwhere::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        ArgwhereInfo _info;                                      \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            ArgwhereInfo info,                                   \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t x_desc);                  \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void **y,                                            \
            size_t *count,                                       \
            const void *x,                                       \
            void *stream) const;                                 \
    };                                                           \
    }

class ArgwhereInfo {
private:
    ArgwhereInfo() = default;

public:
    infiniDtype_t dtype;
    std::vector<ptrdiff_t> strides;
    std::vector<size_t> shapes;
    size_t num_elements;

    static utils::Result<ArgwhereInfo>
    create(
        infiniopTensorDescriptor_t x_desc) {
        CHECK_OR_RETURN(x_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);

        const infiniDtype_t data_type = x_desc->dtype();
        CHECK_DTYPE(data_type, INFINI_DTYPE_F32);

        return utils::Result<ArgwhereInfo>(ArgwhereInfo{
            data_type,
            x_desc->strides(),
            x_desc->shape(),
            x_desc->numel()});
    }
};
#endif
