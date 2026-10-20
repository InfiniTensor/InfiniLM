#ifndef INFINI_RT_TEST_HELPER_H_
#define INFINI_RT_TEST_HELPER_H_

#include <cstddef>
#include <iostream>
#include <string_view>

namespace infini::rt::test {

class TestContext {
 public:
  bool Expect(bool condition, std::string_view message) {
    if (condition) {
      return true;
    }

    std::cerr << message << std::endl;
    failed_ = true;
    return false;
  }

  template <typename Lhs, typename Rhs>
  bool ExpectEqual(const Lhs& lhs, const Rhs& rhs, std::string_view message) {
    return Expect(lhs == rhs, message);
  }

  int ExitCode() const { return failed_ ? 1 : 0; }

 private:
  bool failed_{false};
};

}  // namespace infini::rt::test

#endif
