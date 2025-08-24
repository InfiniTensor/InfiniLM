{
    depfiles = "operator.o: src/infiniop/ops/rope/operator.cc  src/infiniop/ops/rope/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/rope/../../handle.h include/infiniop/handle.h  include/infiniop/ops/rope.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/rope/cpu/rope_cpu.h src/infiniop/ops/rope/cpu/../rope.h  src/infiniop/ops/rope/cpu/../../../../utils.h  src/infiniop/ops/rope/cpu/../../../../utils/custom_types.h  src/infiniop/ops/rope/cpu/../../../../utils/rearrange.h  src/infiniop/ops/rope/cpu/../../../../utils/result.hpp  src/infiniop/ops/rope/cpu/../../../../utils/check.h include/infinicore.h  src/infiniop/ops/rope/cpu/../../../operator.h  src/infiniop/ops/rope/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/rope/cpu/../../../../utils.h\
",
    values = {
        "/usr/bin/gcc",
        {
            "-m64",
            "-fPIC",
            "-O3",
            "-std=c++17",
            "-Iinclude",
            "-DENABLE_CPU_API",
            "-DENABLE_OMP",
            "-DENABLE_CUDNN_API",
            "-finput-charset=UTF-8",
            "-fexec-charset=UTF-8",
            "-DNDEBUG"
        }
    },
    files = {
        "src/infiniop/ops/rope/operator.cc"
    },
    depfiles_format = "gcc"
}