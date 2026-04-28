#include "infinicore/ops/floor.hpp"

namespace infinicore::op {
common::OpDispatcher<Floor::schema> &Floor::dispatcher() {
    static common::OpDispatcher<Floor::schema> dispatcher_;
    return dispatcher_;
};
void Floor::execute(Tensor output, Tensor input) {
    dispatcher().lookup(context::getDevice().getType())(output, input);
}

Tensor floor(Tensor input) {

    auto output = Tensor::empty(input->shape(), input->dtype(), input->device());
    floor_(output, input);
    return output;
}
void floor_(Tensor output, Tensor input) {
    Floor::execute(output, input);
}

} // namespace infinicore::op
