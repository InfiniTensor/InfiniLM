#ifndef INFINI_OPS_TRITON_JIT_COMPILER_H_
#define INFINI_OPS_TRITON_JIT_COMPILER_H_

#include <optional>
#include <string>
#include <vector>

#include "triton/jit/backend.h"
#include "triton/jit/config_.h"

namespace infini::ops::triton::jit {

namespace detail {

struct AutoTuningCandidate {
  Config config;

  Grid grid;

  std::string signature;

  std::string output_prefix;

  std::string cache_identity;
};

std::string CompilationFingerprint(const std::string& op_name);

void Compile(const Target& target, const std::string& op_name,
             const std::string& output_prefix, const Config& config,
             const std::string& signature, const std::string& cache_identity);

std::optional<Config> AutoTune(
    const Target& target, int device_id, const std::string& op_name,
    const std::vector<AutoTuningCandidate>& candidates,
    const std::vector<Argument>& arguments, void* stream,
    int warmup_milliseconds, int repetition_milliseconds);

}  // namespace detail

}  // namespace infini::ops::triton::jit

#endif
