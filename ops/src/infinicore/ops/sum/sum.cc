#include "infinicore/ops/sum.hpp"

#include "../../utils.hpp"
#include <stdexcept>
#include <vector>

namespace infinicore::op {

common::OpDispatcher<Sum::schema> &Sum::dispatcher() {
    static common::OpDispatcher<Sum::schema> dispatcher_;
    return dispatcher_;
};
void Sum::execute(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(input->device());
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Sum implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, dim, keepdim);
}

Tensor sum(Tensor input, std::vector<size_t> dim, bool keepdim) {
    auto in_shape = input->shape();
    std::vector<size_t> out_shape;
    if (dim.empty()) {
        for (size_t i = 0; i < in_shape.size(); i++) {
            dim.push_back(i);
        }
    }
    std::sort(dim.begin(), dim.end());
    if (dim.size() == in_shape.size() && !keepdim) {
        out_shape = {};
    } else {
        if (keepdim) {
            size_t j = 0;
            for (size_t i = 0; i < in_shape.size(); i++) {
                if (j < dim.size() && dim[j] == i) {
                    out_shape.push_back(1);
                    j++;
                } else {
                    out_shape.push_back(in_shape[i]);
                }
            }
        } else {
            size_t j = 0;
            for (size_t i = 0; i < in_shape.size(); i++) {
                if (j < dim.size() && dim[j] == i) {
                    j++;
                } else {
                    out_shape.push_back(in_shape[i]);
                }
            }
        }
    }
    auto output = Tensor::empty(out_shape, input->dtype(), input->device());
    sum_(output, input, dim, keepdim);
    return output;
}

void sum_(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    Sum::execute(output, input, dim, keepdim);
}
} // namespace infinicore::op
