#include "ops.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <chrono>
#include <cmath>
#include <infinirt.h>
#include <iostream>
#include <numeric>

namespace infiniop_test {
std::unordered_map<std::string, const TestBuilder> TEST_BUILDERS = TEST_BUILDER_MAPPINGS;

std::string Result::toString() const {
    std::ostringstream oss;
    oss << "Status: ";
    switch (_status) {
    case TestStatus::PASS:
        oss << GREEN << "PASS" << RESET;
        break;
    case TestStatus::TEST_INIT_FAILED:
        oss << RED << "INVALID TEST" << RESET;
        break;
    case TestStatus::OP_CREATION_FAILED:
        oss << RED << "OP CREATION FAILED" << RESET;
        break;
    case TestStatus::OP_EXECUTION_FAILED:
        oss << RED << "EXECUTION FAILED" << RESET;
        break;
    case TestStatus::RESULT_INCORRECT:
        oss << RED << "WRONG ANSWER" << RESET;
        break;
    default:
        oss << YELLOW << "SKIPPED" << RESET;
        break;
    }
    oss << std::endl;
    oss << "Description: " << _description << std::endl;
    if (_time > 0.) {
        oss << "Time: " << _time << " us" << std::endl;
    } else {
        oss << "Time: N/A" << std::endl;
    }
    if (_error_message.size() > 0) {
        oss << "Error: " << _error_message << std::endl;
    }
    return oss.str();
}

std::vector<std::shared_ptr<Result>> runAllTests(const GGUFFileReader &gguf_reader,
                                                 infiniDevice_t device, int device_id,
                                                 size_t warm_ups, size_t iterations,
                                                 double rtol, double atol) {
    auto meta = gguf_reader.getAttributeMap();
    auto count_meta = meta.find("test_count");
    if (count_meta == meta.end()) {
        throw std::runtime_error("Invalid GGUF file: missing test_count attribute");
    }
    size_t count = *(size_t *)(count_meta->second->value.data());
    std::cout << "Found " << count << " tests" << std::endl;
    auto results = std::vector<std::shared_ptr<Result>>(count);
    try {
        for (size_t i = 0; i < count; i++) {
            results[i] = runTest(gguf_reader, device, device_id, warm_ups, iterations, rtol, atol, i);
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return results;
}

std::shared_ptr<Result> runTest(const GGUFFileReader &gguf_reader,
                                infiniDevice_t device, int device_id,
                                size_t warm_ups, size_t iterations,
                                double rtol, double atol, size_t test_id) {
    auto meta = gguf_reader.getAttributeMap();
    auto tensor_info = gguf_reader.getTensorInfoMap();
    auto name_meta = meta.find("test." + std::to_string(test_id) + ".op_name");
    if (name_meta != meta.end()) {
        std::string op_name(name_meta->second->value.begin(), name_meta->second->value.end());
        auto builder = TEST_BUILDERS.find(op_name)->second;
        auto attrs = std::unordered_map<std::string, std::vector<uint8_t>>();
        auto tensors = std::unordered_map<std::string, std::shared_ptr<Tensor>>();
        infiniopHandle_t handle;
        CHECK_OR(infinirtSetDevice(device, device_id), throw std::runtime_error("Failed to set device"));
        CHECK_OR(infiniopCreateHandle(&handle), throw std::runtime_error("Failed to create handle"));
        for (auto attr_name : builder.attribute_names) {
            auto attr = meta.find("test." + std::to_string(test_id) + "." + attr_name);
            if (attr != meta.end()) {
                attrs[attr_name] = attr->second->value;
            }
        }

        for (auto tensor_name : builder.tensor_names) {
            auto info = tensor_info.find("test." + std::to_string(test_id) + "." + tensor_name);
            if (info != tensor_info.end()) {
                auto shape = meta.find("test." + std::to_string(test_id) + "." + tensor_name + ".shape");
                auto strides = meta.find("test." + std::to_string(test_id) + "." + tensor_name + ".strides");
                bool is_output = std::find(builder.output_names.begin(), builder.output_names.end(), tensor_name) != builder.output_names.end();
                tensors[tensor_name] = std::make_shared<Tensor>(
                    info->second.get(),
                    gguf_reader.getGgmlStart(),
                    shape != meta.end() ? shape->second.get() : nullptr,
                    strides != meta.end() ? strides->second.get() : nullptr,
                    is_output);
            }
        }
        std::shared_ptr<infiniop_test::base::Test> test;
        try {
            test = builder.build(attrs, tensors, rtol, atol);
        } catch (const std::exception &e) {
            return TEST_INIT_FAILED(op_name + "/n" + e.what());
        }

        std::shared_ptr<Result> result;
        try {
            result = test->run(handle, device, device_id, warm_ups, iterations);
        } catch (const std::exception &e) {
            return TEST_INIT_FAILED(op_name + "/n" + e.what());
        }

        CHECK_OR(infiniopDestroyHandle(handle), throw std::runtime_error("Failed to destroy handle"));
        return result;
    }
    return TEST_INIT_FAILED("");
}

void incrementOffset(ptrdiff_t &offset_1, const std::vector<ptrdiff_t> &strides_1, size_t data_size_1,
                     ptrdiff_t &offset_2, const std::vector<ptrdiff_t> &strides_2, size_t data_size_2,
                     std::vector<size_t> &counter, const std::vector<size_t> &shape) {
    for (ptrdiff_t d = shape.size() - 1; d >= 0; d--) {
        counter[d] += 1;
        offset_1 += strides_1[d] * data_size_1;
        offset_2 += strides_2[d] * data_size_2;
        if (counter[d] < shape[d]) {
            break;
        }
        counter[d] = 0;
        offset_1 -= shape[d] * strides_1[d] * data_size_1;
        offset_2 -= shape[d] * strides_2[d] * data_size_2;
    }
}

void allClose(std::shared_ptr<Tensor> actual_, std::shared_ptr<Tensor> expected_, double rtol, double atol) {
    auto actual = actual_->to(INFINI_DEVICE_CPU);
    auto expected = expected_->to(INFINI_DEVICE_CPU);
    auto shape = actual->shape();
    if (shape != expected->shape()) {
        throw std::runtime_error("Shape mismatch.");
    }
    auto ndim = shape.size();
    size_t total = std::accumulate(shape.begin(), shape.end(), (size_t)1, std::multiplies<size_t>());
    auto counter = std::vector<size_t>(ndim, 0);
    ptrdiff_t actual_offset = 0,
              expected_offset = 0;
    size_t num_failed = 0;
    std::string first_failed_msg;
    for (size_t i = 0; i < total; i++) {
        double a_ = getVal((char *)actual->data() + actual_offset, actual->ggml_type());
        double e_ = getVal((char *)expected->data() + expected_offset, expected->ggml_type());
        if (std::fabs(a_ - e_) > atol && std::fabs(a_ - e_) > rtol * std::fmax(std::fabs(a_), std::fabs(e_))) {
            if (num_failed == 0) {
                first_failed_msg = "First failed at index " + std::to_string(i) + " with value " + std::to_string(a_) + " but should be " + std::to_string(e_) + ".";
            }
            num_failed++;
        }
        incrementOffset(actual_offset, actual->strides(), ggmlTypeSize(actual->ggml_type()),
                        expected_offset, expected->strides(), ggmlTypeSize(expected->ggml_type()),
                        counter, shape);
    }
    if (num_failed > 0) {
        throw std::runtime_error(std::to_string(num_failed) + " out of " + std::to_string(total) + " values failed. " + first_failed_msg);
    }
}

double benchmark(std::function<void()> func, size_t warmups, size_t iterations) {
    if (iterations == 0) {
        return 0.0;
    }
    for (size_t i = 0; i < warmups; ++i) {
        func();
    }
    infinirtDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        func();
    }
    infinirtDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double average_time = duration.count() / iterations / 1e3; // average in us

    return average_time;
}
} // namespace infiniop_test
