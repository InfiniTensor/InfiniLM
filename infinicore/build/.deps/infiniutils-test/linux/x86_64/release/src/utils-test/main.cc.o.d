{
    depfiles = "main.o: src/utils-test/main.cc src/utils-test/utils_test.h  src/utils-test/../utils.h src/utils-test/../utils/custom_types.h  src/utils-test/../utils/rearrange.h src/utils-test/../utils/result.hpp  src/utils-test/../utils/check.h include/infinicore.h\
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
            "-DNDEBUG"
        }
    },
    files = {
        "src/utils-test/main.cc"
    },
    depfiles_format = "gcc"
}