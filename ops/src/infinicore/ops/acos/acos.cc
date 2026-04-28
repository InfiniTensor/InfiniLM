#include "infinicore/ops/acos.hpp"

namespace infinicore::op {

common::OpDispatcher<Acos::schema> &Acos::dispatcher() {
    static common::OpDispatcher<Acos::schema> dispatcher_;
    return dispatcher_;
};

void Acos::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

Tensor acos(Tensor input) {
    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    acos_(output, input);
    return output;
}

void acos_(Tensor output, Tensor input) {
    Acos::execute(output, input);
}

} // namespace infinicore::op
