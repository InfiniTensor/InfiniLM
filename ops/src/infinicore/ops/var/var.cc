#include "infinicore/ops/var.hpp"

#include "../../utils.hpp"
#include <stdexcept>
#include <vector>

namespace infinicore::op {

common::OpDispatcher<Var::schema> &Var::dispatcher() {
    static common::OpDispatcher<Var::schema> dispatcher_;
    return dispatcher_;
};

void Var::execute(Tensor var_output, Tensor input, std::vector<size_t> dim, bool unbiased, bool keepdim) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(var_output, input);
    infinicore::context::setDevice(input->device());
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No Var implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(var_output, input, dim, unbiased, keepdim);
}

Tensor var(Tensor input, std::vector<size_t> dim, bool unbiased, bool keepdim) {
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
    auto var_output = Tensor::empty(out_shape, input->dtype(), input->device());
    var_(var_output, input, dim, unbiased, keepdim);
    return var_output;
}

void var_(Tensor var_output, Tensor input, std::vector<size_t> dim, bool unbiased, bool keepdim) {
    Var::execute(var_output, input, dim, unbiased, keepdim);
}
} // namespace infinicore::op
