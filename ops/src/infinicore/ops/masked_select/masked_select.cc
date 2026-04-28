#include <infinicore/ops/masked_select.hpp>
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<MaskedSelect::schema> &MaskedSelect::dispatcher() {
    static common::OpDispatcher<MaskedSelect::schema> dispatcher_;
    return dispatcher_;
};

void MaskedSelect::execute(Tensor input, Tensor mask, void **data_ptr, size_t *dlen_ptr) {
    auto device_type = context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);

    if (func == nullptr) {
        throw std::runtime_error("No MaskedSelect implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    func(input, mask, data_ptr, dlen_ptr);
}

Tensor masked_select(Tensor input, Tensor mask) {

    std::byte *data;
    size_t dlen;
    MaskedSelect::execute(input, mask, (void **)&data, &dlen);

    auto out = Tensor::from_blob(data, {dlen}, input->dtype(), input->device());

    return out;
}

} // namespace infinicore::op
