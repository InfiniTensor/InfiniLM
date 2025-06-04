#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::rope {
struct Test::Attributes {
    std::shared_ptr<Tensor> y;
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> pos_ids;
    std::shared_ptr<Tensor> sin_table;
    std::shared_ptr<Tensor> cos_table;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (tensors.find("y") == tensors.end()
        || tensors.find("x") == tensors.end()
        || tensors.find("pos_ids") == tensors.end()
        || tensors.find("sin_table") == tensors.end()
        || tensors.find("cos_table") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->y = tensors["y"];
    test->_attributes->x = tensors["x"];
    test->_attributes->pos_ids = tensors["pos_ids"];
    test->_attributes->sin_table = tensors["sin_table"];
    test->_attributes->cos_table = tensors["cos_table"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopRoPEDescriptor_t op_desc;
    auto y = _attributes->y->to(device, device_id);
    auto x = _attributes->x->to(device, device_id);
    auto pos_ids = _attributes->pos_ids->to(device, device_id);
    auto sin_table = _attributes->sin_table->to(device, device_id);
    auto cos_table = _attributes->cos_table->to(device, device_id);

    CHECK_OR(infiniopCreateRoPEDescriptor(handle, &op_desc,
                                          y->desc(),
                                          x->desc(),
                                          pos_ids->desc(),
                                          sin_table->desc(),
                                          cos_table->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));

    size_t workspace_size;

    CHECK_OR(infiniopGetRoPEWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    void *workspace;

    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));

    CHECK_OR(infiniopRoPE(op_desc, workspace, workspace_size,
                          y->data(),
                          x->data(),
                          pos_ids->data(),
                          sin_table->data(),
                          cos_table->data(),
                          nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(y, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopRoPE(
                op_desc, workspace, workspace_size,
                y->data(),
                x->data(),
                pos_ids->data(),
                sin_table->data(),
                cos_table->data(),
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"y", "x", "pos_ids", "sin_table", "cos_table", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"y"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- y: " << _attributes->y->info() << std::endl;
    oss << "- x: " << _attributes->x->info() << std::endl;
    oss << "- pos_ids: " << _attributes->pos_ids->info() << std::endl;
    oss << "- sin_table: " << _attributes->sin_table->info() << std::endl;
    oss << "- cos_table: " << _attributes->cos_table->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::rope
