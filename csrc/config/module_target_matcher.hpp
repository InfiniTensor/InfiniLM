#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace infinilm::config {

class ModuleTargetMatcher {
public:
    // Match kinds are ordered from least to most specific.
    enum class MatchKind {
        NONE,
        MODULE_TYPE,
        REGEX,
        EXACT_NAME,
    };

    static MatchKind match(
        std::string_view target,
        std::string_view module_name,
        std::string_view module_type);

    static MatchKind match_any(
        const std::vector<std::string> &targets,
        std::string_view module_name,
        std::string_view module_type);
};

} // namespace infinilm::config
