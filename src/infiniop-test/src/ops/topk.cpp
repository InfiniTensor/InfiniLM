#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include "infiniop/ops/topk.h"
#include <iomanip>
#include <iostream>

namespace infiniop_test::topk {

namespace {
std::string strategy_to_string(int strategy) {
    // This mapping is based on the internal op::topk::TopKStrategy enum.
    // 0: DEEPSEEK_V3
    // 1: STANDARD_SOFTMAX
    switch (strategy) {
    case 0:
        return "DEEPSEEK_V3";
    case 1:
        return "STANDARD_SOFTMAX";
    default:
        return "UNKNOWN";
    }
}
} // namespace

struct Test::Attributes {
    int k;
    int n_group;
    int topk_group;
    int strategy;

    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> ans_val;
    std::shared_ptr<Tensor> ans_ind;
    std::shared_ptr<Tensor> output_val;
    std::shared_ptr<Tensor> output_ind;
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

    test->_attributes->k = *reinterpret_cast<int *>(attributes["k"].data());
    test->_attributes->n_group =
        *reinterpret_cast<int *>(attributes["n_group"].data());
    test->_attributes->topk_group =
        *reinterpret_cast<int *>(attributes["topk_group"].data());
    test->_attributes->strategy =
        *reinterpret_cast<int *>(attributes["strategy"].data());

    test->_attributes->input = tensors["input"];
    test->_attributes->bias = tensors["bias"];
    test->_attributes->ans_val = tensors["ans_val"];
    test->_attributes->ans_ind = tensors["ans_ind"];
    test->_attributes->output_val = tensors["output_val"];
    test->_attributes->output_ind = tensors["output_ind"];

    return test;
}

std::shared_ptr<infiniop_test::Result>
Test::run(infiniopHandle_t handle, infiniDevice_t device, int device_id,
          size_t warm_ups, size_t iterations) {
    infiniopTopKDescriptor_t op_desc;
    auto k = _attributes->k;
    auto n_group = _attributes->n_group;
    auto topk_group = _attributes->topk_group;
    auto strategy = _attributes->strategy;

    auto input = _attributes->input->to(device, device_id);
    auto bias = _attributes->bias->to(device, device_id);
    auto output_val = _attributes->output_val->to(device, device_id);
    auto output_ind = _attributes->output_ind->to(device, device_id);

    CHECK_OR(infiniopCreateTopKDescriptor(
                 handle, &op_desc, input->desc(), output_val->desc(),
                 output_ind->desc(), bias->desc(), k, strategy, n_group, topk_group),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to create op descriptor."));

    size_t workspace_size = infiniopGetTopKWorkspaceSize(op_desc);
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED,
                                "Failed to allocate workspace."));
    CHECK_OR(infiniopTopKCalculate(op_desc, input->data(), output_val->data(),
                                   output_ind->data(), bias->data(), workspace,
                                   nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED,
                                "Failed during execution."));

    try {
        allClose(output_val, _attributes->ans_val, _rtol, _atol);
        allEqual(output_ind, _attributes->ans_ind);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopTopKCalculate(op_desc, input->data(), output_val->data(),
                                  output_ind->data(), bias->data(), workspace,
                                  nullptr);
        },
        warm_ups, iterations);

    infiniopDestroyTopKDescriptor(op_desc);
    infinirtFree(workspace);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"k", "n_group", "topk_group", "strategy"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input",      "bias",       "ans_val",
            "ans_ind",    "output_val", "output_ind"};
}

std::vector<std::string> Test::output_names() {
    return {"output_val", "output_ind"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- k=" << _attributes->k << ", n_group=" << _attributes->n_group
        << ", topk_group=" << _attributes->topk_group
        << ", strategy=" << strategy_to_string(_attributes->strategy)
        << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- bias: " << _attributes->bias->info() << std::endl;
    oss << "- output_val: " << _attributes->output_val->info() << std::endl;
    oss << "- output_ind: " << _attributes->output_ind->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() { delete _attributes; }

} // namespace infiniop_test::topk 