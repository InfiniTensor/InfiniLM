{
    depfiles = "tensor_descriptor.o: src/infiniop/tensor_descriptor.cc  src/infiniop/../utils.h src/infiniop/../utils/custom_types.h  src/infiniop/../utils/rearrange.h src/infiniop/../utils/result.hpp  src/infiniop/../utils/check.h include/infinicore.h src/infiniop/tensor.h  include/infiniop/tensor_descriptor.h include/infiniop/../infinicore.h\
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
        "src/infiniop/tensor_descriptor.cc"
    },
    depfiles_format = "gcc"
}