#ifndef INFINI_OPS_EXAMPLES_RUNTIME_API_H_
#define INFINI_OPS_EXAMPLES_RUNTIME_API_H_

#include "device.h"
#include "runtime.h"

#ifdef WITH_NVIDIA
#include "native/cuda/nvidia/ops/gemm/cublas.h"
#include "native/cuda/nvidia/ops/gemm/cublaslt.h"
#include "native/cuda/nvidia/runtime_.h"
#elif WITH_ILUVATAR
#include "native/cuda/iluvatar/ops/gemm/cublas.h"
#include "native/cuda/iluvatar/runtime_.h"
#elif WITH_METAX
#include "native/cuda/metax/ops/gemm/mcblas.h"
#include "native/cuda/metax/runtime_.h"
#elif WITH_MARS
#include "native/cuda/mars/ops/gemm/hcblas.h"
#include "native/cuda/mars/runtime_.h"
#elif WITH_CAMBRICON
#include "native/cambricon/ops/gemm/cnblas.h"
#include "native/cambricon/runtime_.h"
#elif WITH_MOORE
#include "native/cuda/moore/ops/gemm/mublas.h"
#include "native/cuda/moore/runtime_.h"
#elif WITH_ASCEND
#include "native/ascend/ops/gemm/kernel.h"
#include "native/ascend/runtime_.h"
#elif WITH_CPU
#include <infini/rt/cpu/runtime_.h>

#include "native/cpu/ops/gemm/gemm.h"
#else
#error "One `WITH_*` backend must be enabled for the examples."
#endif

namespace infini::ops {

#ifdef WITH_NVIDIA
using DefaultRuntimeUtils = Runtime<Device::Type::kNvidia>;
#elif WITH_ILUVATAR
using DefaultRuntimeUtils = Runtime<Device::Type::kIluvatar>;
#elif WITH_METAX
using DefaultRuntimeUtils = Runtime<Device::Type::kMetax>;
#elif WITH_MARS
using DefaultRuntimeUtils = Runtime<Device::Type::kMars>;
#elif WITH_CAMBRICON
using DefaultRuntimeUtils = Runtime<Device::Type::kCambricon>;
#elif WITH_MOORE
using DefaultRuntimeUtils = Runtime<Device::Type::kMoore>;
#elif WITH_ASCEND
using DefaultRuntimeUtils = Runtime<Device::Type::kAscend>;
#elif WITH_CPU
using DefaultRuntimeUtils = Runtime<Device::Type::kCpu>;
#endif

}  // namespace infini::ops

#endif
