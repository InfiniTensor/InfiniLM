{
    depfiles = "infiniccl.o: src/infiniccl/infiniccl.cc include/infiniccl.h  include/infinirt.h include/infinicore.h  src/infiniccl/./ascend/infiniccl_ascend.h  src/infiniccl/./ascend/../infiniccl_impl.h  src/infiniccl/./cuda/infiniccl_cuda.h  src/infiniccl/./cuda/../infiniccl_impl.h  src/infiniccl/./metax/infiniccl_metax.h  src/infiniccl/./metax/../infiniccl_impl.h\
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
        "src/infiniccl/infiniccl.cc"
    },
    depfiles_format = "gcc"
}