#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::random_sample {
struct Test::Attributes {
    float random_val;
    float topp;
    int topk;
    int voc;
    float temperature;

    std::shared_ptr<Tensor> data;
    std::shared_ptr<Tensor> ans;
    std::shared_ptr<Tensor> result;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    if (attributes.find("random_val") == attributes.end()
        || attributes.find("topp") == attributes.end()
        || attributes.find("topk") == attributes.end()
        || attributes.find("voc") == attributes.end()
        || attributes.find("temperature") == attributes.end()
        || tensors.find("data") == tensors.end()
        || tensors.find("ans") == tensors.end()
        || tensors.find("result") == tensors.end()) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->random_val = *reinterpret_cast<float *>(attributes["random_val"].data());
    test->_attributes->topp = *reinterpret_cast<float *>(attributes["topp"].data());
    test->_attributes->topk = *reinterpret_cast<int *>(attributes["topk"].data());
    test->_attributes->voc = *reinterpret_cast<int *>(attributes["voc"].data());
    test->_attributes->temperature = *reinterpret_cast<float *>(attributes["temperature"].data());

    test->_attributes->data = tensors["data"];
    test->_attributes->ans = tensors["ans"];
    test->_attributes->result = tensors["result"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    infiniopRandomSampleDescriptor_t op_desc;
    auto random_val = _attributes->random_val;
    auto topp = _attributes->topp;
    auto topk = _attributes->topk;
    auto temperature = _attributes->temperature;
    auto data = _attributes->data->to(device, device_id);
    auto result = _attributes->result->to(device, device_id);
    CHECK_OR(infiniopCreateRandomSampleDescriptor(handle, &op_desc,
                                                  result->desc(),
                                                  data->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));
    size_t workspace_size;
    CHECK_OR(infiniopGetRandomSampleWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    CHECK_OR(infiniopRandomSample(op_desc, workspace, workspace_size,
                                  result->data(),
                                  data->data(),
                                  random_val,
                                  topp,
                                  topk,
                                  temperature,
                                  nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    try {
        allClose(result, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopRandomSample(
                op_desc, workspace, workspace_size,
                result->data(),
                data->data(),
                random_val,
                topp,
                topk,
                temperature,
                nullptr);
        },
        warm_ups, iterations);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"random_val", "topp", "topk", "voc", "temperature"};
}

std::vector<std::string> Test::tensor_names() {
    return {"data", "ans", "result"};
}

std::vector<std::string> Test::output_names() {
    return {"result"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- random_val=" << _attributes->random_val
        << ", topp=" << _attributes->topp << std::endl
        << ", topk=" << _attributes->topk << std::endl
        << ", voc=" << _attributes->voc << std::endl
        << ", temperature=" << _attributes->temperature << std::endl;
    oss << "- data: " << _attributes->data->info() << std::endl;
    oss << "- result: " << _attributes->result->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::random_sample
