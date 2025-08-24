{
    depfiles = "clip_cpu.o: src/infiniop/ops/clip/cpu/clip_cpu.cc  src/infiniop/ops/clip/cpu/clip_cpu.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/elementwise_cpu.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../devices/cpu/common_cpu.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/custom_types.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/rearrange.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/result.hpp  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/check.h  include/infinicore.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../devices/cpu/cpu_handle.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../devices/cpu/../../handle.h  include/infiniop/handle.h include/infiniop/../infinicore.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../elementwise.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../../utils.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/clip/cpu/../../../elementwise/cpu/../../../utils.h  include/infiniop/ops/clip.h  include/infiniop/ops/../operator_descriptor.h\
",
    values = {
        "/usr/bin/gcc",
        {
            "-m64",
            "-fvisibility=hidden",
            "-fvisibility-inlines-hidden",
            "-Wall",
            "-Werror",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-fPIC",
            "-Wno-unknown-pragmas",
            "-fopenmp",
            "-DNDEBUG"
        }
    },
    files = {
        "src/infiniop/ops/clip/cpu/clip_cpu.cc"
    },
    depfiles_format = "gcc"
}