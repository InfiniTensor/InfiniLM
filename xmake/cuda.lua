
local CUDA_ROOT = os.getenv("CUDA_ROOT") or os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH")
local CUDNN_ROOT = os.getenv("CUDNN_ROOT") or os.getenv("CUDNN_HOME") or os.getenv("CUDNN_PATH")
if CUDA_ROOT ~= nil then
    add_includedirs(CUDA_ROOT .. "/include")
end
if CUDNN_ROOT ~= nil then
    add_includedirs(CUDNN_ROOT .. "/include")
end

target("infiniop-cuda")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cublas", "cudnn")
    add_cugencodes("native")

    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cuflags("-Xcompiler=/W3", "-Xcompiler=/WX")
        add_cxxflags("/FS")
        if CUDNN_ROOT ~= nil then
            add_linkdirs(CUDNN_ROOT .. "\\lib\\x64")
        end
    else
        add_cuflags("-Xcompiler=-Wall", "-Xcompiler=-Werror")
        add_cuflags("-Xcompiler=-fPIC")
        add_cuflags("--extended-lambda")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxxflags("-fPIC")
    end

    set_languages("cxx17")
    add_files("../src/infiniop/devices/cuda/*.cu", "../src/infiniop/ops/*/cuda/*.cu")
target_end()

target("infinirt-cuda")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart")

    if is_plat("windows") then
        add_cuflags("-Xcompiler=/utf-8", "--expt-relaxed-constexpr", "--allow-unsupported-compiler")
        add_cxxflags("/FS")
    else
        add_cuflags("-Xcompiler=-fPIC")
        add_culdflags("-Xcompiler=-fPIC")
        add_cxflags("-fPIC")
    end

    set_languages("cxx17")
    add_files("../src/infinirt/cuda/*.cu")
target_end()

target("infiniccl-cuda")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)
    if has_config("ccl") then
        set_policy("build.cuda.devlink", true)
        set_toolchains("cuda")
        add_links("cudart")

        if not is_plat("windows") then
            add_cuflags("-Xcompiler=-fPIC")
            add_culdflags("-Xcompiler=-fPIC")
            add_cxflags("-fPIC")

            local nccl_root = os.getenv("NCCL_ROOT")
            if nccl_root then
                add_includedirs(nccl_root .. "/include")
                add_links(nccl_root .. "/lib/libnccl.so")
            else
                add_links("nccl") -- Fall back to default nccl linking
            end

            add_files("../src/infiniccl/cuda/*.cu")
        else
            print("[Warning] NCCL is not supported on Windows")
        end
    end
    set_languages("cxx17")
    
target_end()
