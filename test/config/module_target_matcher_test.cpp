#include "csrc/config/module_target_matcher.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using MatchKind = infinilm::config::ModuleTargetMatcher::MatchKind;
using infinilm::config::ModuleTargetMatcher;

constexpr std::string_view MODULE_NAME = "model.layers.3.self_attn.q_proj";
constexpr std::string_view MODULE_TYPE = "Linear";

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_exact_name_matching() {
    expect(
        ModuleTargetMatcher::match(MODULE_NAME, MODULE_NAME, MODULE_TYPE)
            == MatchKind::EXACT_NAME,
        "full module name did not match exactly");
    expect(
        ModuleTargetMatcher::match("q_proj", MODULE_NAME, MODULE_TYPE)
            == MatchKind::NONE,
        "module name suffix was treated as an exact match");
}

void test_module_type_matching() {
    expect(
        ModuleTargetMatcher::match("Linear", MODULE_NAME, MODULE_TYPE)
            == MatchKind::MODULE_TYPE,
        "module type did not match");
    expect(
        ModuleTargetMatcher::match("Embedding", MODULE_NAME, MODULE_TYPE)
            == MatchKind::NONE,
        "unrelated module type matched");
}

void test_regex_matching_starts_at_module_name() {
    expect(
        ModuleTargetMatcher::match(
            R"(re:model\.layers\.\d+\.self_attn\..*_proj$)",
            MODULE_NAME,
            MODULE_TYPE)
            == MatchKind::REGEX,
        "regular expression did not match the module name");
    expect(
        ModuleTargetMatcher::match("re:q_proj$", MODULE_NAME, MODULE_TYPE)
            == MatchKind::NONE,
        "regular expression searched beyond the start of the module name");
    expect(
        ModuleTargetMatcher::match("re:.*q_proj$", MODULE_NAME, MODULE_TYPE)
            == MatchKind::REGEX,
        "explicit suffix regular expression did not match");
    expect(
        ModuleTargetMatcher::match("re:model\\.layers", MODULE_NAME, MODULE_TYPE)
            == MatchKind::REGEX,
        "regular expression was incorrectly required to match the full name");
}

void test_match_any_returns_the_most_specific_match() {
    expect(
        ModuleTargetMatcher::match_any(
            {"Linear", "re:model\\.layers", std::string(MODULE_NAME)},
            MODULE_NAME,
            MODULE_TYPE)
            == MatchKind::EXACT_NAME,
        "exact name did not take priority");
    expect(
        ModuleTargetMatcher::match_any(
            {"Linear", "re:model\\.layers"}, MODULE_NAME, MODULE_TYPE)
            == MatchKind::REGEX,
        "regular expression did not take priority over module type");
    expect(
        ModuleTargetMatcher::match_any({}, MODULE_NAME, MODULE_TYPE)
            == MatchKind::NONE,
        "empty target list matched");
}

void test_invalid_regex_is_rejected() {
    try {
        ModuleTargetMatcher::match("re:[", MODULE_NAME, MODULE_TYPE);
    } catch (const std::invalid_argument &error) {
        expect(
            std::string(error.what()).find("re:[") != std::string::npos,
            "invalid-regex error did not identify the target");
        return;
    }
    throw std::runtime_error("invalid regular expression was accepted");
}

} // namespace

int main() {
    try {
        test_exact_name_matching();
        test_module_type_matching();
        test_regex_matching_starts_at_module_name();
        test_match_any_returns_the_most_specific_match();
        test_invalid_regex_is_rejected();
    } catch (const std::exception &error) {
        std::cerr << "module_target_matcher_test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "module_target_matcher_test passed\n";
    return 0;
}
