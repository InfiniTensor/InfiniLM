{
    depfiles = "operator.o: src/infiniop/ops/linear_backwards/operator.cc  src/infiniop/ops/linear_backwards/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/linear_backwards/../../handle.h  include/infiniop/handle.h include/infiniop/ops/linear_backwards.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/linear_backwards/cpu/linear_backwards_cpu.h  src/infiniop/ops/linear_backwards/cpu/../linear_backwards.h  src/infiniop/ops/linear_backwards/cpu/../../../operator.h  src/infiniop/ops/linear_backwards/cpu/../info.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils/custom_types.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils/rearrange.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils/result.hpp  src/infiniop/ops/linear_backwards/cpu/../../../../utils/check.h  include/infinicore.h  src/infiniop/ops/linear_backwards/cpu/../../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/linear_backwards/cpu/../../../../utils.h\
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
        "src/infiniop/ops/linear_backwards/operator.cc"
    },
    depfiles_format = "gcc"
}