#include "execution/optional_submit.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <base/add.h>

#include <array>
#include <stdexcept>
#include <string>

namespace {
struct MissingSdkOperator {};
static_assert(!infinilm::execution::HasSdkFactory<MissingSdkOperator>::value);
static_assert(infinilm::execution::HasSdkFactory<infini::ops::Add>::value);
} // namespace

int main() {
    using namespace infinilm;
    const runtime::Device device{runtime::Device::Type::kCpu, 0};
    auto rt = runtime::Runtime::create(device);
    std::array<float, 4> a{1.0f, -2.0f, 0.0f, 4.0f};
    std::array<float, 4> b{2.0f, 3.0f, -4.0f, 0.5f};
    std::array<float, 4> out{};
    auto wrap = [&](auto &v) {
        return runtime::Tensor::from_blob(v.data(), {4}, runtime::DataType::kFloat32, device);
    };
    auto x = wrap(a), y = wrap(b), z = wrap(out);
    execution::submit_optional<infini::ops::Add>(
        "Add", *rt, execution::RecordingPolicy::DeviceOnly,
        infini::ops::Config::Automatic(), x, y, 2.0, z);
    rt->syncStream();
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] != a[i] + 2.0f * b[i]) {
            throw std::runtime_error("Available SDK operator result mismatch.");
        }
    }
    auto expected_config = infini::ops::Add::DefaultConfig(device.type());
    auto optional_config = execution::optional_default_config<infini::ops::Add>("Add", device.type());
    if (expected_config.implementation_index() != optional_config.implementation_index()) {
        throw std::runtime_error("Optional SDK configuration changed provider selection.");
    }
    bool config_rejected = false;
    try {
        execution::optional_default_config<MissingSdkOperator>("MissingSdkOperator", device.type());
    } catch (const std::runtime_error &e) {
        config_rejected = std::string(e.what()).find("MissingSdkOperator") != std::string::npos;
    }
    if (!config_rejected) {
        throw std::runtime_error("Missing SDK configuration was not rejected.");
    }
    if (!execution::has_sdk_implementation<infini::ops::Add>(device.type())
        || execution::has_sdk_implementation<MissingSdkOperator>(device.type())) {
        throw std::runtime_error("Optional SDK provider availability mismatch.");
    }
    bool rejected = false;
    try {
        execution::submit_optional<MissingSdkOperator>(
            "MissingSdkOperator", *rt, execution::RecordingPolicy::DeviceOnly,
            infini::ops::Config::Automatic());
    } catch (const std::runtime_error &e) {
        rejected = std::string(e.what()).find("MissingSdkOperator") != std::string::npos;
    }
    if (!rejected) {
        throw std::runtime_error("Missing SDK binding was not rejected.");
    }
    return 0;
}
