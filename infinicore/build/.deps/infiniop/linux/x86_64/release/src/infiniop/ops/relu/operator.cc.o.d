{
    depfiles = "operator.o: src/infiniop/ops/relu/operator.cc  src/infiniop/ops/relu/../../operator.h  include/infiniop/operator_descriptor.h include/infiniop/handle.h  include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h  src/infiniop/ops/relu/../../handle.h include/infiniop/handle.h  include/infiniop/ops/relu.h  include/infiniop/ops/../operator_descriptor.h  src/infiniop/ops/relu/cpu/relu_cpu.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/elementwise_cpu.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../devices/cpu/common_cpu.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/custom_types.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/rearrange.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/result.hpp  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../devices/cpu/../../../utils/check.h  include/infinicore.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../devices/cpu/cpu_handle.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../devices/cpu/../../handle.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../elementwise.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../../utils.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../operator.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../tensor.h  include/infiniop/tensor_descriptor.h  src/infiniop/ops/relu/cpu/../../../elementwise/cpu/../../../utils.h\
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
        "src/infiniop/ops/relu/operator.cc"
    },
    depfiles_format = "gcc"
}