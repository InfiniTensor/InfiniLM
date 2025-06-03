#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::mul {
struct Test::Attributes {
    std::shared_ptr<Tensor> a;
    std::shared_ptr<Tensor> b;
    std::shared_ptr<Tensor> c;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("a") == tensors.end()
        || tensors.find("b") == tensors.end()
        || tensors.find("c") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->a = tensors["a"];
    test->_attributes->b = tensors["b"];
    test->_attributes->c = tensors["c"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopMulDescriptor_t op_desc;
    auto a = _attributes->a->to(device, device_id);
    auto b = _attributes->b->to(device, device_id);
    auto c = _attributes->c->to(device, device_id);
    CHECK_OR(infiniopCreateMulDescriptor(handle, &op_desc,
                                         c->desc(),
                                         a->desc(),
                                         b->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetMulWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopMul(op_desc, workspace, workspace_size,
                         c->data(),
                         a->data(),
                         b->data(),
                         nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(c, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopMul(
                op_desc, workspace, workspace_size,
                c->data(),
                a->data(),
                b->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"a", "b", "c", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"c"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- a: " << _attributes->a->info() << std::endl;
    oss << "- b: " << _attributes->b->info() << std::endl;
    oss << "- c: " << _attributes->c->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::mul
