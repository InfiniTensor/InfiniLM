#include "infinicore/ops/topk.hpp"

#include "../../utils.hpp"
#include <stdexcept>
#include <vector>

namespace infinicore::op {

common::OpDispatcher<TopK::schema> &TopK::dispatcher() {
    static common::OpDispatcher<TopK::schema> dispatcher_;
    return dispatcher_;
};
void TopK::execute(Tensor values_output, Tensor indices_output, Tensor input, size_t k, size_t dim, bool largest, bool sorted) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(values_output, input);
    infinicore::context::setDevice(input->device());
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Topk implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(values_output, indices_output, input, k, dim, largest, sorted);
}

std::pair<Tensor, Tensor> topk(Tensor input, size_t k, size_t dim, bool largest, bool sorted) {
    auto in_shape = input->shape();
    std::vector<size_t> out_shape = in_shape;
    out_shape[dim] = k;

    auto values_output = Tensor::empty(out_shape, input->dtype(), input->device());
    auto indices_output = Tensor::empty(out_shape, DataType::I32, input->device());
    topk_(values_output, indices_output, input, k, dim, largest, sorted);
    return {values_output, indices_output};
}

void topk_(Tensor values_output, Tensor indices_output, Tensor input, size_t k, size_t dim, bool largest, bool sorted) {
    TopK::execute(values_output, indices_output, input, k, dim, largest, sorted);
}
} // namespace infinicore::op
