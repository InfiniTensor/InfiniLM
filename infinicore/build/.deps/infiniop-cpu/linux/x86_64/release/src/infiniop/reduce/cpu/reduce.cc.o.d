{
    depfiles = "reduce.o: src/infiniop/reduce/cpu/reduce.cc  src/infiniop/reduce/cpu/reduce.h  src/infiniop/reduce/cpu/../../../utils.h  src/infiniop/reduce/cpu/../../../utils/custom_types.h  src/infiniop/reduce/cpu/../../../utils/rearrange.h  src/infiniop/reduce/cpu/../../../utils/result.hpp  src/infiniop/reduce/cpu/../../../utils/check.h include/infinicore.h\
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
        "src/infiniop/reduce/cpu/reduce.cc"
    },
    depfiles_format = "gcc"
}