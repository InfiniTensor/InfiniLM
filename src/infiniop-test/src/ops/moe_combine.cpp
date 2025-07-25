#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::moe_combine {
struct Test::Attributes {
    std::shared_ptr<Tensor> permuted_input;
    std::shared_ptr<Tensor> gating_weights;
    std::shared_ptr<Tensor> aux_info;
    std::shared_ptr<Tensor> output;
    std::shared_ptr<Tensor> ans;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (!check_names(attributes, Test::attribute_names()) ||
        !check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->permuted_input = tensors["permuted_input"];
    test->_attributes->gating_weights = tensors["gating_weights"];
    test->_attributes->aux_info = tensors["aux_info"];
    test->_attributes->output = tensors["output"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result>
Test::run(infiniopHandle_t handle, infiniDevice_t device, int device_id,
          size_t warm_ups, size_t iterations) {
    infiniopMoECombineDescriptor_t op_desc;

    auto permuted_input = _attributes->permuted_input->to(device, device_id);
    auto gating_weights = _attributes->gating_weights->to(device, device_id);
    auto aux_info = _attributes->aux_info->to(device, device_id);
    auto output = _attributes->output->to(device, device_id);

    CHECK_OR(infiniopCreateMoECombineDescriptor(
                 handle, &op_desc, permuted_input->desc(),
                 gating_weights->desc(), aux_info->desc(), output->desc()),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to create op descriptor."));

    CHECK_OR(infiniopMoECombine(op_desc, output->data(),
                                permuted_input->data(),
                                gating_weights->data(), aux_info->data(),
                                nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED,
                                "Failed during execution."));

    try {
        allClose(output, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopMoECombine(op_desc, output->data(), permuted_input->data(),
                               gating_weights->data(), aux_info->data(),
                               nullptr);
        },
        warm_ups, iterations);

    infiniopDestroyMoECombineDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() { return {}; }

std::vector<std::string> Test::tensor_names() {
    return {"permuted_input", "gating_weights", "aux_info", "output", "ans"};
}

std::vector<std::string> Test::output_names() { return {"output"}; }

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- permuted_input: " << _attributes->permuted_input->info()
        << std::endl;
    oss << "- gating_weights: " << _attributes->gating_weights->info()
        << std::endl;
    oss << "- aux_info: " << _attributes->aux_info->info() << std::endl;
    oss << "- output: " << _attributes->output->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() { delete _attributes; }

} // namespace infiniop_test::moe_combine 