{
    depfiles = "linear_backwards_cpu.o:  src/infiniop/ops/linear_backwards/cpu/linear_backwards_cpu.cc  src/infiniop/ops/linear_backwards/cpu/linear_backwards_cpu.h  src/infiniop/ops/linear_backwards/cpu/../linear_backwards.h  src/infiniop/ops/linear_backwards/cpu/../../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/linear_backwards/cpu/../info.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils/custom_types.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils/rearrange.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils/result.hpp  src/infiniop/ops/linear_backwards/cpu/../../../../utils/check.h  include/infinicore.h  src/infiniop/ops/linear_backwards/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils.h  src/infiniop/ops/linear_backwards/cpu/../../../devices/cpu/common_cpu.h  src/infiniop/ops/linear_backwards/cpu/../../../devices/cpu/../../../utils.h  src/infiniop/ops/linear_backwards/cpu/../../../devices/cpu/cpu_handle.h  src/infiniop/ops/linear_backwards/cpu/../../../devices/cpu/../../handle.h  include/infiniop/handle.h\
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
        "src/infiniop/ops/linear_backwards/cpu/linear_backwards_cpu.cc"
    },
    depfiles_format = "gcc"
}