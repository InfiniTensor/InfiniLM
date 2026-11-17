#include "infinicore/ops/silu.hpp"
#include "infinicore/graph/graph.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {
namespace {

// Keep the existing backend dispatcher, but defer execution during graph
// recording just like the planned operators that produce and consume SiLU.
class RecordedSilu final : public graph::GraphOperator {
public:
    RecordedSilu(const Tensor &output, const Tensor &input, Silu::schema function)
        : output_(output), input_(input), function_(function) {}

    void run() const override {
        context::setDevice(output_->device());
        function_(output_, input_);
    }

private:
    graph::GraphTensor output_, input_;
    Silu::schema function_;
};

} // namespace

common::OpDispatcher<Silu::schema> &Silu::dispatcher() {
    static common::OpDispatcher<Silu::schema> dispatcher_;
    return dispatcher_;
};

void Silu::execute(Tensor output, Tensor input) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(output->device());
    auto device_type = output->device().type();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Silu implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    if (context::isGraphRecording()) {
        context::addGraphOperator(std::make_shared<RecordedSilu>(output, input, func));
    } else {
        func(output, input);
    }
}

Tensor silu(Tensor input) {
    Shape shape = input->shape();
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    silu_(output, input);
    return output;
}

void silu_(Tensor output, Tensor input) {
    Silu::execute(output, input);
}
} // namespace infinicore::op
