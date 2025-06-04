#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::clip {
struct Test::Attributes {
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> min_val;
    std::shared_ptr<Tensor> max_val;
    std::shared_ptr<Tensor> y;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (tensors.find("x") == tensors.end()
        || tensors.find("min_val") == tensors.end()
        || tensors.find("max_val") == tensors.end()
        || tensors.find("y") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->x = tensors["x"];
    test->_attributes->min_val = tensors["min_val"];
    test->_attributes->max_val = tensors["max_val"];
    test->_attributes->y = tensors["y"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopClipDescriptor_t op_desc;
    auto x = _attributes->x->to(device, device_id);
    auto min_val = _attributes->min_val->to(device, device_id);
    auto max_val = _attributes->max_val->to(device, device_id);
    auto y = _attributes->y->to(device, device_id);
    CHECK_OR(infiniopCreateClipDescriptor(handle, &op_desc,
                                          y->desc(),
                                          x->desc(),
                                          min_val->desc(),
                                          max_val->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create clip descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetClipWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopClip(op_desc, workspace, workspace_size,
                          y->data(),
                          x->data(),
                          min_val->data(),
                          max_val->data(),
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
            infiniopClip(
                op_desc, workspace, workspace_size,
                y->data(),
                x->data(),
                min_val->data(),
                max_val->data(),
                nullptr);
        },
        warm_ups, iterations);

    infiniopDestroyClipDescriptor(op_desc);
    infinirtFree(workspace);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"x", "min_val", "max_val", "y", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"y"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- x: " << _attributes->x->info() << std::endl;
    oss << "- min_val: " << _attributes->min_val->info() << std::endl;
    oss << "- max_val: " << _attributes->max_val->info() << std::endl;
    oss << "- y: " << _attributes->y->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::clip
