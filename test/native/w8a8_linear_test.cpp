#include "execution/graph/graph.hpp"
#include "layers/quantization/compressed_tensors.hpp"
#include "runtime/runtime.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace infinilm;
using runtime::DataType;
using runtime::Tensor;

int main() {
    const runtime::Device device{runtime::Device::Type::kHygon, 0};
    auto rt = runtime::Runtime::create(device);
    const auto config = nlohmann::json::parse(R"({"format":"int-quantized","config_groups":{"group0":{"weights":{"num_bits":8,"type":"int","strategy":"channel","symmetric":true,"dynamic":false},"input_activations":{"num_bits":8,"type":"int","strategy":"token","symmetric":true,"dynamic":true}}}})");
    quantization::CompressedTensors layer(config);
    std::string phase = "initial";
    size_t eager_checks = 0, graph_checks = 0, device_graphs = 0;
    for (const auto shape : {std::array<size_t, 3>{1, 17, 1024}, {17, 33, 256}, {65, 65, 64}, {257, 257, 64}}) {
        const auto m = shape[0], n = shape[1], k = shape[2];
        std::vector<float> x(m * k), scales(n), bias(n), actual(m * n);
        std::vector<int8_t> weights(n * k);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = static_cast<int8_t>(int((i * 37 + 13) % 256) - 128);
        }
        for (size_t c = 0; c < n; ++c) {
            scales[c] = float(c % 8 + 1) / 1024.0f;
            bias[c] = float(int(c % 17) - 8) / 16.0f;
        }
        auto input = Tensor::empty(*rt, {1, m, k}, DataType::kFloat32);
        quantization::ParamsMap params{{"weight", Tensor::empty(*rt, {n, k}, DataType::kInt8)},
                                       {"weight_scale", Tensor::empty(*rt, {n, 1}, DataType::kFloat32)},
                                       {"bias", Tensor::empty(*rt, {n}, DataType::kFloat32)}};
        rt->memcpyH2D(params.at("weight")->data(), weights.data(), weights.size());
        rt->memcpyH2D(params.at("weight_scale")->data(), scales.data(), scales.size() * sizeof(float));
        rt->memcpyH2D(params.at("bias")->data(), bias.data(), bias.size() * sizeof(float));
        rt->syncStream();
        auto populate = [&](int seed) {
            for (size_t row = 0; row < m; ++row) {
                const float step = float(1u << (row % 3));
                for (size_t j = 0; j < k; ++j) {
                    x[row * k + j] = float(int((j * 13 + row + seed * 7) % 253) - 126) * 0.5f * step;
                }
                x[row * k] = 127.0f * step;
                if (seed == 2) {
                    std::fill(x.begin() + row * k, x.begin() + (row + 1) * k, 0.0f);
                }
            }
            rt->memcpyH2D(input->data(), x.data(), x.size() * sizeof(float));
            rt->syncStream();
        };
        auto check = [&](const Tensor &output, bool has_bias, float alpha) {
            if (output->shape() != runtime::Shape{1, m, n}) {
                throw std::runtime_error("W8A8 output shape mismatch");
            }
            rt->memcpyD2H(actual.data(), output->data(), actual.size() * sizeof(float));
            rt->syncStream();
            for (size_t row = 0; row < m; ++row) {
                float maximum = 1.0e-10f;
                for (size_t j = 0; j < k; ++j) {
                    maximum = std::max(maximum, std::fabs(x[row * k + j]));
                }
                volatile float scale = maximum / 127.0f;
                volatile float inverse = 127.0f / maximum;
                for (size_t col = 0; col < n; ++col) {
                    int64_t sum = 0;
                    for (size_t j = 0; j < k; ++j) {
                        volatile float scaled = x[row * k + j] * inverse;
                        const int q = std::clamp(int(std::nearbyint(scaled)), -127, 127);
                        sum += int64_t(q) * weights[col * k + j];
                    }
                    volatile float weight_scale = scales[col] * alpha;
                    volatile float joint = scale * weight_scale;
                    volatile float product = float(sum) * joint;
                    float expected = has_bias ? product + bias[col] : product;
                    if (std::memcmp(&expected, &actual[row * n + col], sizeof(float)) != 0) {
                        std::cerr << nlohmann::json{{"passed", false}, {"phase", phase}, {"shape", {m, n, k}}, {"row", row}, {"column", col}, {"alpha", alpha}, {"bias", has_bias}, {"expected", expected}, {"actual", actual[row * n + col]}, {"eager_checks", eager_checks}, {"graph_checks", graph_checks}}.dump() << "\n";
                        throw std::runtime_error("W8A8 layer differs from independent integer oracle");
                    }
                }
            }
        };
        for (bool has_bias : {false, true}) {
            for (float alpha : {1.0f, 0.5f}) {
                populate(0);
                phase = "eager";
                auto eager = layer.forward(*rt, params, input, has_bias, alpha);
                check(eager, has_bias, alpha);
                ++eager_checks;
                rt->startGraphRecording(alpha == 1.0f);
                Tensor output;
                std::shared_ptr<execution::graph::Graph> graph;
                try {
                    output = layer.forward(*rt, params, input, has_bias, alpha);
                    graph = rt->stopGraphRecording();
                } catch (...) {
                    rt->cancelGraphRecording();
                    throw;
                }
                const auto report = graph->execution_report();
                if (alpha == 1.0f) {
                    if (report.device_graph_segments == 0 || report.host_segments != 0) {
                        throw std::runtime_error("W8A8 required a device graph but used host replay");
                    }
                    ++device_graphs;
                }
                for (int replay = 0; replay < 3; ++replay) {
                    populate(replay);
                    // Exercise temporary allocation reuse while captured quant buffers remain owned.
                    {
                        auto scratch = Tensor::empty(*rt, {m, k}, DataType::kFloat32);
                        rt->setDeviceMemory(scratch->data(), 0, m * k * sizeof(float));
                    }
                    phase = "graph_replay_" + std::to_string(replay);
                    graph->run();
                    check(output, has_bias, alpha);
                    ++graph_checks;
                }
            }
        }
    }
    std::cout << "{\"passed\":true,\"eager_checks\":" << eager_checks
              << ",\"graph_checks\":" << graph_checks << ",\"device_graphs\":" << device_graphs << "}\n";
}
