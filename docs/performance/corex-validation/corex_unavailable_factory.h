#include <infini/ops.h>
#include <stdexcept>
namespace infini::ops::detail {
// Private common CoreX build adapter. Available generated specializations win.
// An unavailable provider remains ineligible and throws if explicitly invoked.
template<typename Key> struct OperatorFactory {
 template<typename... Args> static std::unique_ptr<Operator<Key>> Make(const Config&, Args&&...) {
  throw std::invalid_argument("operator unavailable for this CoreX build");
 }
 template<typename... Args> static void Call(const Handle&, const Config&, const Args&...) {
  throw std::invalid_argument("operator unavailable for this CoreX build");
 }
 template<typename... Args> static bool CanMake(const Config&, const Args&...) { return false; }
};
}
