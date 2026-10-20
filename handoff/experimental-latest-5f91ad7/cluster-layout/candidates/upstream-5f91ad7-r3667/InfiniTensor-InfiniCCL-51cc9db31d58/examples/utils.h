#ifndef INFINI_CCL_EXAMPLES_UTILS_H_
#define INFINI_CCL_EXAMPLES_UTILS_H_

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Simple check macro for the C-API.
#define CHECK_INFINI(cmd)                                                   \
  do {                                                                      \
    infinicclResult_t res = (cmd);                                          \
    if (res != infinicclSuccess) {                                          \
      std::cerr << "[InfiniCCL Error] example program received error code " \
                << res << " at line " << __LINE__ << std::endl;             \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

#define CHECK_RT(runtime_type, cmd) \
  CHECK_INFINI(static_cast<infinicclResult_t>(runtime_type::Check(cmd)))

// Simple Timer for Profiling
class Timer {
  std::chrono::high_resolution_clock::time_point start;

 public:
  Timer() : start(std::chrono::high_resolution_clock::now()) {}
  double ElapsedMs() const {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
  }
};

struct Metrics {
  double elapsed_ms;
  size_t total_bytes;
  int world_size;

  void Print() const {
    double seconds = elapsed_ms / 1000.0;
    double gigabytes =
        static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);

    // Industry standard formula: `2 * (n-1) / n`.
    double bus_bw =
        (2.0 * (world_size - 1) / world_size) * (gigabytes / seconds);
    double alg_bw = gigabytes / seconds;

    std::cout << "Time:           " << std::fixed << std::setprecision(3)
              << elapsed_ms << " ms" << std::endl;
    std::cout << "Throughput:     " << std::fixed << std::setprecision(2)
              << bus_bw << " GB/s (Bus BW)" << std::endl;
    std::cout << "Alg Bandwidth:  " << std::fixed << std::setprecision(2)
              << alg_bw << " GB/s" << std::endl;
    return;
  }
};

class Validator {
 public:
  template <typename T>
  static bool ValidateResult(const T *data, size_t count, T expected_val,
                             int rank, bool verbose = false,
                             std::string op_name = "\b") {
    bool correct = true;
    int error_count = 0;

    for (size_t i = 0; i < count; ++i) {
      if (std::fabs(static_cast<double>(data[i]) -
                    static_cast<double>(expected_val)) > 1e-3) {
        correct = false;
        error_count++;
        if (error_count <= 3 && rank == 0) {
          std::cerr << "Error at index " << i << ": " << data[i]
                    << " != " << expected_val << std::endl;
        }
      }
    }

    if (verbose && rank == 0) {
      const char *GREEN = "\033[32m";
      const char *RED = "\033[31m";
      const char *RESET = "\033[0m";

      std::cout << "\n=== " << op_name << " Results ===" << std::endl;
      std::cout << "Correct: "
                << (correct ? (GREEN + std::string("YES") + RESET)
                            : (RED + std::string("NO") + RESET));
      if (!correct) {
        std::cout << " (" << error_count << " errors)";
      }
      std::cout << std::endl;

      std::cout << "Expect:  " << std::fixed << std::setprecision(2)
                << static_cast<double>(expected_val) << std::endl;
      std::cout << "Actual:  " << std::fixed << std::setprecision(2)
                << static_cast<double>(data[0]) << std::endl;
    }
    return correct;
  }
};

#endif
