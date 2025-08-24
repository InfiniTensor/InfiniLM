{
    depfiles = "handle.o: src/infiniop/devices/handle.cc include/infiniop/handle.h  include/infiniop/../infinicore.h src/infiniop/devices/../../utils.h  src/infiniop/devices/../../utils/custom_types.h  src/infiniop/devices/../../utils/rearrange.h  src/infiniop/devices/../../utils/result.hpp  src/infiniop/devices/../../utils/check.h include/infinicore.h  include/infinirt.h include/infinicore.h  src/infiniop/devices/cpu/cpu_handle.h  src/infiniop/devices/cpu/../../handle.h\
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
        "src/infiniop/devices/handle.cc"
    },
    depfiles_format = "gcc"
}