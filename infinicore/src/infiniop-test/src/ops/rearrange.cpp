#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::rearrange {
struct Test::Attributes {
    std::shared_ptr<Tensor> dst, src, ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (!check_names(attributes, Test::attribute_names()) || !check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->dst = tensors["dst"];
    test->_attributes->src = tensors["src"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle,
    infiniDevice_t device,
    int device_id,
    size_t warm_ups,
    size_t iterations) {

    infiniopRearrangeDescriptor_t op_desc;
    auto dst = _attributes->dst->to(device, device_id);
    auto src = _attributes->src->to(device, device_id);
    CHECK_OR(infiniopCreateRearrangeDescriptor(
                 handle, &op_desc,
                 dst->desc(),
                 src->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    CHECK_OR(infiniopRearrange(
                 op_desc,
                 dst->data(),
                 src->data(),
                 nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allEqual(dst, _attributes->ans);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopRearrange(
                op_desc,
                dst->data(),
                src->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"dst", "src", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"dst"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl
        << "- dst: " << _attributes->dst->info() << std::endl
        << "- src: " << _attributes->src->info() << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::rearrange
