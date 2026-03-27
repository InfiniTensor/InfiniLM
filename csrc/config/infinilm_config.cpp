#include "infinilm_config.hpp"
#include <stdexcept>

namespace infinilm::config {

namespace {

thread_local std::shared_ptr<InfinilmConfig> _current_infinilm_config;

} // namespace

void set_current_infinilm_config(const std::shared_ptr<InfinilmConfig> &config) {
    assert(nullptr == _current_infinilm_config);
    assert(nullptr != config);
    _current_infinilm_config = config;
}

const InfinilmConfig &get_current_infinilm_config() {
    assert(nullptr != _current_infinilm_config && "Current Infinilm config is not set.");
    return *_current_infinilm_config;
}

} // namespace infinilm::config
