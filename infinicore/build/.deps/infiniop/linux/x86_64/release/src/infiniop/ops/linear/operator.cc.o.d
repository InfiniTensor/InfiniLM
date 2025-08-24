{
    depfiles = "operator.o: src/infiniop/ops/linear/operator.cc  src/infiniop/ops/linear/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/linear/../../handle.h include/infiniop/handle.h  include/infiniop/ops/linear.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/linear/cpu/linear_cpu.h  src/infiniop/ops/linear/cpu/../linear.h  src/infiniop/ops/linear/cpu/../../../operator.h  src/infiniop/ops/linear/cpu/../info.h  src/infiniop/ops/linear/cpu/../../../../utils.h  src/infiniop/ops/linear/cpu/../../../../utils/custom_types.h  src/infiniop/ops/linear/cpu/../../../../utils/rearrange.h  src/infiniop/ops/linear/cpu/../../../../utils/result.hpp  src/infiniop/ops/linear/cpu/../../../../utils/check.h  include/infinicore.h src/infiniop/ops/linear/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/linear/cpu/../../../../utils.h\
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
        "src/infiniop/ops/linear/operator.cc"
    },
    depfiles_format = "gcc"
}