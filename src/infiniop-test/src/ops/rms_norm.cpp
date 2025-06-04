#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::rms_norm {
struct Test::Attributes {
    float epsilon;
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> ans;
    std::shared_ptr<Tensor> y;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (attributes.find("epsilon") == attributes.end()
        || tensors.find("x") == tensors.end()
        || tensors.find("w") == tensors.end()
        || tensors.find("ans") == tensors.end()
        || tensors.find("y") == tensors.end()) {
        throw std::runtime_error("Invalid Test: Missing attributes or tensors");
    }

    test->_attributes->epsilon = *reinterpret_cast<float *>(attributes["epsilon"].data());

    test->_attributes->x = tensors["x"];
    test->_attributes->w = tensors["w"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->y = tensors["y"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopRMSNormDescriptor_t op_desc;
    CHECK_OR(infiniopCreateRMSNormDescriptor(handle, &op_desc,
                                             _attributes->y->desc(),
                                             _attributes->x->desc(),
                                             _attributes->w->desc(),
                                             _attributes->epsilon),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create RMSNorm descriptor"));

    auto x = _attributes->x->to(device, device_id);
    auto w = _attributes->w->to(device, device_id);
    auto y = _attributes->y->to(device, device_id);

    size_t workspace_size;
    CHECK_OR(infiniopGetRMSNormWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size"));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace"));
    }

    CHECK_OR(infiniopRMSNorm(op_desc,
                             workspace, workspace_size,
                             y->data(),
                             x->data(),
                             w->data(),
                             nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "RMSNorm execution failed"));

    try {
        allClose(y, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopRMSNorm(op_desc,
                            workspace, workspace_size,
                            y->data(),
                            x->data(),
                            w->data(),
                            nullptr);
        },
        warm_ups, iterations);

    if (workspace != nullptr) {
        infinirtFree(workspace);
    }

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"epsilon"};
}

std::vector<std::string> Test::tensor_names() {
    return {"x", "w", "ans", "y"};
}

std::vector<std::string> Test::output_names() {
    return {"y"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- epsilon=" << _attributes->epsilon << std::endl;
    oss << "- x: " << _attributes->x->info() << std::endl;
    oss << "- w: " << _attributes->w->info() << std::endl;
    oss << "- y: " << _attributes->y->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::rms_norm
