#pragma once

#include "linear.hpp"
#include "runtime/runtime.hpp"

#include <memory>
#include <vector>

namespace infinilm::nn {

// Prepared Eager RowParallelLinear tasks on one Runtime/TP2 communicator.
// submit() binds current inputs by copying them into owned staging buffers.
// Outputs are borrowed until the next submit. Consumers use this Runtime's
// stream and are submitted before complete(), which is the completion boundary.
// Other-stream input producers must be completed before submit(). Rebuild the
// plan after repacking/replacing weights or changing the layer configuration.
// Construct, submit, complete and destroy on the Runtime's owning thread.
class RowBatchPlan {
public:
    using Layers = std::vector<std::shared_ptr<RowParallelLinear>>;
    using Tensors = std::vector<infinilm::runtime::Tensor>;
    enum class State { Ready,
                       InFlight,
                       Failed };

    RowBatchPlan(std::shared_ptr<infinilm::runtime::Runtime> runtime,
                 Layers layers, const Tensors &input_layouts);
    ~RowBatchPlan() noexcept;
    RowBatchPlan(const RowBatchPlan &) = delete;
    RowBatchPlan &operator=(const RowBatchPlan &) = delete;

    const Tensors &submit(const Tensors &inputs);
    void complete();
    State state() const noexcept { return state_; }

private:
    struct PreparedGemm;
    void check_context() const;
    void check_inputs(const Tensors &inputs) const;

    std::shared_ptr<infinilm::runtime::Runtime> runtime_;
    Layers layers_;
    std::vector<std::unique_ptr<PreparedGemm>> gemms_;
    Tensors outputs_, in_flight_inputs_;
    std::vector<const void *> sends_;
    std::vector<void *> receives_;
    std::vector<size_t> counts_;
    infinicclComm_t communicator_ = nullptr;
    infinicclDataType_t ccl_dtype_{};
    State state_ = State::Ready;
};

} // namespace infinilm::nn
