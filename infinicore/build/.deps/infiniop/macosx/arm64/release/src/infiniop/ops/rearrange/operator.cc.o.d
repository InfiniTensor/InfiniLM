{
    values = {
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang",
        {
            "-Qunused-arguments",
            "-target",
            "arm64-apple-macos15.4",
            "-isysroot",
            "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.5.sdk",
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
    depfiles_format = "gcc",
    depfiles = "build/.objs/infiniop/macosx/arm64/release/src/infiniop/ops/rearrange/__cpp_operator.cc.cc:   src/infiniop/ops/rearrange/operator.cc   src/infiniop/ops/rearrange/../../operator.h   include/infiniop/operator_descriptor.h include/infiniop/handle.h   include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h   src/infiniop/ops/rearrange/../../handle.h   include/infiniop/ops/rearrange.h   include/infiniop/ops/../operator_descriptor.h   src/infiniop/ops/rearrange/cpu/rearrange_cpu.h   src/infiniop/ops/rearrange/cpu/../rearrange.h   src/infiniop/ops/rearrange/cpu/../../../../utils.h   src/infiniop/ops/rearrange/cpu/../../../../utils/custom_types.h   src/infiniop/ops/rearrange/cpu/../../../../utils/rearrange.h   src/infiniop/ops/rearrange/cpu/../../../../utils/result.hpp   src/infiniop/ops/rearrange/cpu/../../../../utils/check.h   include/infinicore.h   src/infiniop/ops/rearrange/cpu/../../../operator.h\
",
    files = {
        "src/infiniop/ops/rearrange/operator.cc"
    }
}