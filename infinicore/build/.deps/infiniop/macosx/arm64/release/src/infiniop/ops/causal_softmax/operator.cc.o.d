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
    depfiles = "build/.objs/infiniop/macosx/arm64/release/src/infiniop/ops/causal_softmax/__cpp_operator.cc.cc:   src/infiniop/ops/causal_softmax/operator.cc   src/infiniop/ops/causal_softmax/../../operator.h   include/infiniop/operator_descriptor.h include/infiniop/handle.h   include/infiniop/../infinicore.h include/infiniop/tensor_descriptor.h   src/infiniop/ops/causal_softmax/../../handle.h   include/infiniop/ops/causal_softmax.h   include/infiniop/ops/../operator_descriptor.h   src/infiniop/ops/causal_softmax/cpu/causal_softmax_cpu.h   src/infiniop/ops/causal_softmax/cpu/../causal_softmax.h   src/infiniop/ops/causal_softmax/cpu/../../../operator.h   src/infiniop/ops/causal_softmax/cpu/../info.h   src/infiniop/ops/causal_softmax/cpu/../../../../utils.h   src/infiniop/ops/causal_softmax/cpu/../../../../utils/custom_types.h   src/infiniop/ops/causal_softmax/cpu/../../../../utils/rearrange.h   src/infiniop/ops/causal_softmax/cpu/../../../../utils/result.hpp   src/infiniop/ops/causal_softmax/cpu/../../../../utils/check.h   include/infinicore.h   src/infiniop/ops/causal_softmax/cpu/../../../tensor.h\
",
    files = {
        "src/infiniop/ops/causal_softmax/operator.cc"
    }
}