#include "csrc/layers/quantization/compressed_tensors.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using infinilm::quantization::CompressedTensors;
using infinilm::quantization::ParamDescriptor;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

const ParamDescriptor &find_descriptor(
    const std::vector<ParamDescriptor> &layout,
    const std::string &name) {
    for (const auto &descriptor : layout) {
        if (descriptor.name == name) {
            return descriptor;
        }
    }
    throw std::runtime_error("missing parameter descriptor: " + name);
}

void expect_partition(
    const ParamDescriptor &descriptor,
    int split_dim,
    int tp_rank,
    int tp_size) {
    expect(
        descriptor.split_dim == split_dim,
        descriptor.name + " has an incorrect split dimension");
    expect(
        descriptor.tp_rank == tp_rank,
        descriptor.name + " has an incorrect TP rank");
    expect(
        descriptor.tp_size == tp_size,
        descriptor.name + " has an incorrect TP size");
}

void test_replicated_layout() {
    const CompressedTensors quantization(nlohmann::json::object());
    const auto layout = quantization.get_param_layout(
        128, 256, -1, 0, 1, -1, infinicore::DataType::F32, true);

    expect(layout.size() == 3, "replicated layout has an incorrect parameter count");
    const auto &weight = find_descriptor(layout, "weight");
    const auto &scale = find_descriptor(layout, "weight_scale");
    const auto &bias = find_descriptor(layout, "bias");
    expect(weight.shape == std::vector<size_t>{256, 128}, "weight shape is incorrect");
    expect(weight.dtype == infinicore::DataType::I8, "weight dtype is not INT8");
    expect(scale.shape == std::vector<size_t>{256, 1}, "weight-scale shape is incorrect");
    expect(scale.dtype == infinicore::DataType::F32, "weight-scale dtype is not FP32");
    expect_partition(weight, -1, 0, 1);
    expect_partition(scale, -1, 0, 1);
    expect_partition(bias, -1, 0, 1);
}

void test_column_parallel_layout() {
    const CompressedTensors quantization(nlohmann::json::object());
    const auto layout = quantization.get_param_layout(
        128, 256, 0, 1, 2, -1, infinicore::DataType::F32, true);

    expect_partition(find_descriptor(layout, "weight"), 0, 1, 2);
    expect_partition(find_descriptor(layout, "weight_scale"), 0, 1, 2);
    expect_partition(find_descriptor(layout, "bias"), 0, 1, 2);
}

void test_row_parallel_layout() {
    const CompressedTensors quantization(nlohmann::json::object());
    const auto layout = quantization.get_param_layout(
        128, 256, 1, 1, 2, -1, infinicore::DataType::F32, true);

    expect_partition(find_descriptor(layout, "weight"), 1, 1, 2);
    expect_partition(find_descriptor(layout, "weight_scale"), -1, 0, 1);
    expect_partition(find_descriptor(layout, "bias"), -1, 0, 1);
}

} // namespace

int main() {
    try {
        test_replicated_layout();
        test_column_parallel_layout();
        test_row_parallel_layout();
    } catch (const std::exception &error) {
        std::cerr << "compressed_tensors_test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "compressed_tensors_test passed\n";
    return 0;
}
