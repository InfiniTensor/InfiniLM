#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::moe_dispatch {
struct Test::Attributes {
    int num_experts;

    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> indices;
    std::shared_ptr<Tensor> permuted_output;
    std::shared_ptr<Tensor> aux_info;
    std::shared_ptr<Tensor> ans_permuted_output;
    std::shared_ptr<Tensor> ans_aux_info;
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

    test->_attributes->num_experts =
        *reinterpret_cast<int *>(attributes["num_experts"].data());

    test->_attributes->input = tensors["input"];
    test->_attributes->indices = tensors["indices"];
    test->_attributes->permuted_output = tensors["permuted_output"];
    test->_attributes->aux_info = tensors["aux_info"];
    test->_attributes->ans_permuted_output = tensors["ans_permuted_output"];
    test->_attributes->ans_aux_info = tensors["ans_aux_info"];

    return test;
}

std::shared_ptr<infiniop_test::Result>
Test::run(infiniopHandle_t handle, infiniDevice_t device, int device_id,
          size_t warm_ups, size_t iterations) {
    infiniopMoEDispatchDescriptor_t op_desc;
    auto num_experts = _attributes->num_experts;

    auto input = _attributes->input->to(device, device_id);
    auto indices = _attributes->indices->to(device, device_id);
    auto permuted_output = _attributes->permuted_output->to(device, device_id);
    auto aux_info = _attributes->aux_info->to(device, device_id);

    CHECK_OR(infiniopCreateMoEDispatchDescriptor(
                 handle, &op_desc, num_experts, input->desc(), indices->desc(),
                 permuted_output->desc(), aux_info->desc()),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to create op descriptor."));

    CHECK_OR(infiniopMoEDispatch(op_desc, permuted_output->data(),
                                 aux_info->data(), input->data(),
                                 indices->data(), nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED,
                                "Failed during execution."));

    try {
        allClose(permuted_output, _attributes->ans_permuted_output, _rtol,
                 _atol);
        allEqual(aux_info, _attributes->ans_aux_info);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopMoEDispatch(op_desc, permuted_output->data(),
                                aux_info->data(), input->data(),
                                indices->data(), nullptr);
        },
        warm_ups, iterations);

    infiniopDestroyMoEDispatchDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() { return {"num_experts"}; }

std::vector<std::string> Test::tensor_names() {
    return {"input",       "indices",           "permuted_output", "aux_info",
            "ans_permuted_output", "ans_aux_info"};
}

std::vector<std::string> Test::output_names() {
    return {"permuted_output", "aux_info"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- num_experts=" << _attributes->num_experts << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- indices: " << _attributes->indices->info() << std::endl;
    oss << "- permuted_output: " << _attributes->permuted_output->info()
        << std::endl;
    oss << "- aux_info: " << _attributes->aux_info->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() { delete _attributes; }

} // namespace infiniop_test::moe_dispatch 