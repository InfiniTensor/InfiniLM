#pragma once

#include "workspace_context.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/tensor.hpp"

#include <string_view>

namespace infinilm::engine {

inline infinicore::Tensor allocate_inference_tensor(
    std::string_view tag,
    const infinicore::Shape &shape,
    infinicore::DataType dtype,
    const infinicore::Device &device,
    WorkspaceZeroPolicy zero_policy = WorkspaceZeroPolicy::None) {
    if (infinicore::context::isGraphRecording()) {
        if (zero_policy == WorkspaceZeroPolicy::OnAcquire || zero_policy == WorkspaceZeroPolicy::OnCreate) {
            return infinicore::Tensor::zeros(shape, dtype, device);
        }
        return infinicore::Tensor::empty(shape, dtype, device);
    }

    (void)tag;
    // Do not pool returned activation tensors here. InfiniCore view/as_strided creates
    // a new TensorImpl that shares storage, but InfiniLM cannot observe that storage
    // lifetime yet, so reusing these buffers can race with surviving views.
    if (zero_policy == WorkspaceZeroPolicy::OnAcquire || zero_policy == WorkspaceZeroPolicy::OnCreate) {
        return infinicore::Tensor::zeros(shape, dtype, device);
    }
    return infinicore::Tensor::empty(shape, dtype, device);
}

} // namespace infinilm::engine
