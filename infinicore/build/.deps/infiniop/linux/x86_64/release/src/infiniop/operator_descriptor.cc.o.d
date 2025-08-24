{
    depfiles = "operator_descriptor.o: src/infiniop/operator_descriptor.cc  src/infiniop/operator.h include/infiniop/operator_descriptor.h  include/infiniop/handle.h include/infiniop/../infinicore.h  include/infiniop/tensor_descriptor.h\
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
        "src/infiniop/operator_descriptor.cc"
    },
    depfiles_format = "gcc"
}