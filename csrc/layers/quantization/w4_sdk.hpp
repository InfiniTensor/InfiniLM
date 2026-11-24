#pragma once

#include "execution/optional_submit.hpp"
#include <base/w4_marlin_gemm.h>
#include <base/w4_marlin_parameters.h>
#include <base/w4_marlin_repack.h>

namespace infinilm::quantization::w4_detail {

template <class Op, class... Args>
void submit(infinilm::runtime::Runtime &runtime, const Args &...args) {
    infinilm::execution::submit_optional<Op>(
        "W4 Marlin", runtime, infinilm::execution::RecordingPolicy::HostOnly,
        infini::ops::Config::Automatic(), args...);
}

} // namespace infinilm::quantization::w4_detail
