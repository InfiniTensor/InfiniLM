#ifndef INFINI_CCL_UTILS_H_
#define INFINI_CCL_UTILS_H_

#include <iostream>

#include "constexpr_map.h"

// Internal helper macros to count arguments.
#define GET_LOG_MACRO(_1, _2, NAME, ...) NAME

#define LOG_1(msg)                 \
  ::infini::ccl::Logger::PrintMsg( \
      (msg), ::infini::ccl::Logger::LogLevel::kError, __FILE__, __LINE__)

#define LOG_2(msg, level) \
  ::infini::ccl::Logger::PrintMsg((msg), (level), __FILE__, __LINE__)

// Temporary logging macro. To be replaced by `glog` in the future.
#define LOG(...) GET_LOG_MACRO(__VA_ARGS__, LOG_2, LOG_1)(__VA_ARGS__)

namespace infini::ccl {

class Logger {
 public:
  enum class LogLevel : int8_t { kInfo, kWarning, kError, kFatal, kCount };

  static void PrintMsg(const char *msg, LogLevel level = LogLevel::kError,
                       const char *file = __FILE__, int line = __LINE__) {
    std::cerr << "[InfiniCCL " << kLogLevelToDesc.at(level) << "] " << file
              << ":" << line << " - " << msg << std::endl;
    if (level == LogLevel::kFatal) {
      std::abort();
    }
  }

 private:
  static constexpr ConstexprMap<LogLevel, std::string_view,
                                static_cast<std::size_t>(LogLevel::kCount)>
      kLogLevelToDesc{{{
          {LogLevel::kInfo, "INFO"},
          {LogLevel::kWarning, "WARNING"},
          {LogLevel::kError, "ERROR"},
          {LogLevel::kFatal, "FATAL"},
      }}};
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_UTILS_H_
