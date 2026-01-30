/**
 * @file fusion_context.cpp
 * @brief Implementation of FusionContext
 */

#include "fusion_context.hpp"

namespace infinilm::fusion {

// Thread-local storage definition
thread_local std::unordered_map<std::string, bool> FusionContext::decisions_;

void FusionContext::set(const std::string &op_name, bool should_fuse) {
    decisions_[op_name] = should_fuse;
}

bool FusionContext::get(const std::string &op_name, bool default_value) {
    auto it = decisions_.find(op_name);
    if (it != decisions_.end()) {
        return it->second;
    }
    return default_value;
}

bool FusionContext::has(const std::string &op_name) {
    return decisions_.find(op_name) != decisions_.end();
}

void FusionContext::clear() {
    decisions_.clear();
}

size_t FusionContext::size() {
    return decisions_.size();
}

} // namespace infinilm::fusion
