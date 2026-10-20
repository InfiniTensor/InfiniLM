#ifndef INFINI_RT_TESTS_PERFORMANCE_PERF_COMMON_H_
#define INFINI_RT_TESTS_PERFORMANCE_PERF_COMMON_H_

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#define INFINI_RT_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define INFINI_RT_NOINLINE __attribute__((noinline))
#else
#define INFINI_RT_NOINLINE
#endif

#ifndef INFINI_RT_PERF_BACKEND_NAME
#define INFINI_RT_PERF_BACKEND_NAME "unknown"
#endif

namespace infini::rt::perf {

struct Param {
  std::string name;
  std::string value;
  bool quoted;
};

inline Param NumberParam(std::string name, std::uint64_t value) {
  return {std::move(name), std::to_string(value), false};
}

inline Param StringParam(std::string name, std::string value) {
  return {std::move(name), std::move(value), true};
}

inline std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (const auto ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << ch;
        break;
    }
  }
  return out.str();
}

template <typename T>
inline void DoNotOptimize(const T& value) {
#if defined(__GNUC__) || defined(__clang__)
  if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) <= sizeof(void*)) {
    asm volatile("" : : "g"(value) : "memory");
  } else {
    asm volatile("" : : "g"(&value) : "memory");
  }
#else
  static volatile const void* sink;
  sink = &value;
#endif
}

inline double Mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }

  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

inline double Median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }

  std::sort(values.begin(), values.end());
  const auto mid = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[mid - 1] + values[mid]) / 2.0;
  }

  return values[mid];
}

inline double ConvertNanoseconds(double ns, const std::string& unit) {
  if (unit == "ns") {
    return ns;
  }
  if (unit == "us") {
    return ns / 1000.0;
  }
  if (unit == "ms") {
    return ns / 1000000.0;
  }

  return ns;
}

inline void PrintResult(const std::string& benchmark,
                        const std::vector<Param>& params,
                        std::size_t iterations, const std::string& unit,
                        double mean, double median) {
  std::cout << std::setprecision(12);
  std::cout << "{\"backend\":\"" << JsonEscape(INFINI_RT_PERF_BACKEND_NAME)
            << "\",\"benchmark\":\"" << JsonEscape(benchmark)
            << "\",\"params\":{";

  for (std::size_t i = 0; i < params.size(); ++i) {
    if (i != 0) {
      std::cout << ",";
    }
    std::cout << "\"" << JsonEscape(params[i].name) << "\":";
    if (params[i].quoted) {
      std::cout << "\"" << JsonEscape(params[i].value) << "\"";
    } else {
      std::cout << params[i].value;
    }
  }

  std::cout << "},\"iterations\":" << iterations << ",\"unit\":\""
            << JsonEscape(unit) << "\",\"mean\":" << mean
            << ",\"median\":" << median << "}" << std::endl;
}

template <typename Func>
void RunBenchmark(const std::string& benchmark,
                  const std::vector<Param>& params, std::size_t iterations,
                  const std::string& unit, Func&& func) {
  constexpr std::size_t kSamples = 7;
  const auto warmup_iterations = std::min<std::size_t>(iterations, 1000);

  for (std::size_t i = 0; i < warmup_iterations; ++i) {
    func();
  }

  std::vector<double> samples;
  samples.reserve(kSamples);

  for (std::size_t sample = 0; sample < kSamples; ++sample) {
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
      func();
    }
    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_ns =
        std::chrono::duration<double, std::nano>(end - start).count();
    samples.push_back(
        ConvertNanoseconds(elapsed_ns / static_cast<double>(iterations), unit));
  }

  PrintResult(benchmark, params, iterations, unit, Mean(samples),
              Median(samples));
}

inline void SkipBenchmark(const std::string& benchmark,
                          const std::string& reason) {
  std::cerr << benchmark << " skipped: " << reason << std::endl;
}

}  // namespace infini::rt::perf

#endif
