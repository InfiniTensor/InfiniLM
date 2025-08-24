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
    depfiles = "build/.objs/infiniop/macosx/arm64/release/src/infiniop/devices/__cpp_handle.cc.cc:   src/infiniop/devices/handle.cc include/infiniop/handle.h   include/infiniop/../infinicore.h src/infiniop/devices/../../utils.h   src/infiniop/devices/../../utils/custom_types.h   src/infiniop/devices/../../utils/rearrange.h   src/infiniop/devices/../../utils/result.hpp   src/infiniop/devices/../../utils/check.h include/infinicore.h   include/infinirt.h src/infiniop/devices/cpu/cpu_handle.h   src/infiniop/devices/cpu/../../handle.h\
",
    files = {
        "src/infiniop/devices/handle.cc"
    }
}