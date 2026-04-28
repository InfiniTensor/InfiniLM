#include "infinicore/ops/all.hpp"

#include "../../utils.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
namespace infinicore::op {

common::OpDispatcher<All::schema> &All::dispatcher() {
    static common::OpDispatcher<All::schema> dispatcher_;
    return dispatcher_;
};
void All::execute(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    infinicore::context::setDevice(input->device());
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No All implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(output, input, dim, keepdim);
}

Tensor all(Tensor input, std::vector<size_t> dim, bool keepdim) {
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
    auto output = Tensor::empty(out_shape, DataType::BOOL, input->device());
    all_(output, input, dim, keepdim);
    return output;
}

void all_(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    All::execute(output, input, dim, keepdim);
}
} // namespace infinicore::op
