#ifndef __GEMM_MOORE_H__
#define __GEMM_MOORE_H__

#include "mublas/gemm_mublas.h"
#include "mudnn/gemm_mudnn.h"

namespace op::gemm::moore {

// Descriptor class for GEMM operations on Moore devices.
// This class acts as a wrapper to select either mublas or mudnn backend.
// It encapsulates the backend-specific Descriptor implementation and provides
// a unified interface for workspace query and GEMM calculation.
class Descriptor final : public InfiniopDescriptor {
public:
    // Destructor: deletes the backend-specific descriptor.
    ~Descriptor() {
        if (_backend == Backend::MUBLAS) {
            delete reinterpret_cast<mublas::Descriptor *>(_impl);
        } else {
            delete reinterpret_cast<mudnn::Descriptor *>(_impl);
        }
    }

    // Returns the required workspace size for the GEMM operation.
    size_t workspaceSize() const {
        if (_backend == Backend::MUBLAS) {
            return reinterpret_cast<mublas::Descriptor *>(_impl)->workspaceSize();
        } else {
            return reinterpret_cast<mudnn::Descriptor *>(_impl)->workspaceSize();
        }
    }

    // Static factory method to create a Descriptor instance.
    // This method chooses the backend (mublas or mudnn) and constructs
    // the corresponding implementation internally.
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc) {
        auto desc = new Descriptor(handle->device, handle->device_id);

        // Backend selection strategy:
        // Currently defaulting to MUDNN.
        // Can be modified to choose based on environment variables or runtime parameters.
        desc->_backend = Backend::MUDNN;

        if (desc->_backend == Backend::MUBLAS) {
            mublas::Descriptor *impl;
            auto status = mublas::Descriptor::create(handle, &impl, c_desc, a_desc, b_desc);
            if (status != INFINI_STATUS_SUCCESS) {
                delete desc;
                return status;
            }
            desc->_impl = impl;
        } else {
            mudnn::Descriptor *impl;
            auto status = mudnn::Descriptor::create(handle, &impl, c_desc, a_desc, b_desc);
            if (status != INFINI_STATUS_SUCCESS) {
                delete desc;
                return status;
            }
            desc->_impl = impl;
        }

        *desc_ptr = desc;
        return INFINI_STATUS_SUCCESS;
    }

    // Unified GEMM calculation interface.
    // Calls the corresponding backend's calculate function internally.
    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *c, float beta,
        const void *a, const void *b,
        float alpha,
        void *stream) const {
        if (_backend == Backend::MUBLAS) {
            return reinterpret_cast<mublas::Descriptor *>(_impl)
                ->calculate(workspace, workspace_size, c, beta, a, b, alpha, stream);
        } else {
            return reinterpret_cast<mudnn::Descriptor *>(_impl)
                ->calculate(workspace, workspace_size, c, beta, a, b, alpha, stream);
        }
    }

private:
    // Private constructor: ensures users cannot directly instantiate Descriptor.
    // Instances must be created via the static create() factory method.
    Descriptor(infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id}, _impl(nullptr) {}

    // Enum to indicate which backend is being used internally.
    enum class Backend { MUBLAS,
                         MUDNN };

    Backend _backend; // Currently selected MUBLAS/MUDNN backend
    void *_impl;      // Pointer to backend-specific descriptor (mublas::Descriptor* or mudnn::Descriptor*)
};

} // namespace op::gemm::moore

#endif // __GEMM_MOORE_H__
