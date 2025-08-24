{
    depfiles = "infinirt_cpu.o: src/infinirt/cpu/infinirt_cpu.cc  src/infinirt/cpu/infinirt_cpu.h src/infinirt/cpu/../infinirt_impl.h  include/infinirt.h include/infinicore.h\
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
            "-fopenmp",
            "-fPIC",
            "-DNDEBUG"
        }
    },
    files = {
        "src/infinirt/cpu/infinirt_cpu.cc"
    },
    depfiles_format = "gcc"
}