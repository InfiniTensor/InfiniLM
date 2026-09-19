#include "module_target_matcher.hpp"

#include <regex>
#include <stdexcept>

namespace infinilm::config {
namespace {

constexpr std::string_view REGEX_PREFIX = "re:";

bool is_regex_target(std::string_view target) {
    return target.size() >= REGEX_PREFIX.size()
        && target.substr(0, REGEX_PREFIX.size()) == REGEX_PREFIX;
}

} // namespace

ModuleTargetMatcher::MatchKind ModuleTargetMatcher::match(
    std::string_view target,
    std::string_view module_name,
    std::string_view module_type) {
    if (target == module_name) {
        return MatchKind::EXACT_NAME;
    }

    if (is_regex_target(target)) {
        const std::string pattern(target.substr(REGEX_PREFIX.size()));
        try {
            const std::regex expression(
                pattern,
                std::regex_constants::ECMAScript | std::regex_constants::optimize);
            if (std::regex_search(
                    module_name.begin(),
                    module_name.end(),
                    expression,
                    std::regex_constants::match_continuous)) {
                return MatchKind::REGEX;
            }
        } catch (const std::regex_error &error) {
            throw std::invalid_argument(
                "invalid module target regex `" + std::string(target)
                + "`: " + error.what());
        }
        return MatchKind::NONE;
    }

    if (target == module_type) {
        return MatchKind::MODULE_TYPE;
    }
    return MatchKind::NONE;
}

ModuleTargetMatcher::MatchKind ModuleTargetMatcher::match_any(
    const std::vector<std::string> &targets,
    std::string_view module_name,
    std::string_view module_type) {
    MatchKind best_match = MatchKind::NONE;
    for (const auto &target : targets) {
        const auto current_match = match(target, module_name, module_type);
        if (current_match > best_match) {
            best_match = current_match;
            if (best_match == MatchKind::EXACT_NAME) {
                break;
            }
        }
    }
    return best_match;
}

} // namespace infinilm::config
