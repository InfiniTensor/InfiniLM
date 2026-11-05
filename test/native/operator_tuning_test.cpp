#include "runtime/runtime.hpp"

#include <tuning.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("operator tuning assertion failed");
    }
}

} // namespace

int main() {
    const auto cache_path = std::filesystem::temp_directory_path() / "infinilm_operator_tuning_test.json";
    std::error_code error;
    std::filesystem::remove(cache_path, error);
    require(!error);

    auto &manager = infini::ops::TuningManager::Instance();
    manager.LoadTuningCache(cache_path.string());
    require(manager.IsEnabled());

    const auto runtime = infinilm::runtime::Runtime::create(
        {infinilm::runtime::Device::Type::kCpu, 0});
    require(runtime != nullptr);

    // Runtime creation must not replace an explicit host-side tuning
    // configuration with an empty environment-derived configuration.
    require(manager.IsEnabled());

    std::filesystem::remove(cache_path, error);
    require(!error);

    std::puts("operator_tuning_test: PASS");
    return 0;
}
