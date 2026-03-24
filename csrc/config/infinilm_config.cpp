#include "infinilm_config.hpp"

namespace infinilm::config {

namespace {

thread_local InfinilmConfig _current_infinilm_config;

} // namespace

void set_current_infinilm_config(const InfinilmConfig &config) {
    _current_infinilm_config = config;
}

const InfinilmConfig &get_current_infinilm_config() {
    return _current_infinilm_config;
}

} // namespace infinilm::config
