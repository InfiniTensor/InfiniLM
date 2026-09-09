#include "infinicore/ops/argwhere.hpp"
#include "../../utils.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"
#include <cstddef>

namespace infinicore::op {

common::OpDispatcher<Argwhere::schema> &Argwhere::dispatcher() {
    static common::OpDispatcher<Argwhere::schema> dispatcher_;
    return dispatcher_;
}

void Argwhere::execute(void **y, size_t *count, Tensor x) {
    auto device_type = context::getDevice().type();

    auto func = dispatcher().lookup(device_type);
    if (func == nullptr) {
        throw std::runtime_error("Argwhere op not implemented for device type " + std::to_string(static_cast<int>(device_type)));
    }
    func(y, count, x);
}
Tensor argwhere(Tensor x) {
    void *y = nullptr;
    size_t count = 0;
    Argwhere::execute(&y, &count, x);
    auto result = Tensor::from_blob(y, Shape{count, x->ndim()}, DataType::kInt64, Device(Device::Type::kCpu));
    result = result->to(x->device());
    return result;
}

} // namespace infinicore::op
