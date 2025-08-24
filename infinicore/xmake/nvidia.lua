local CUDNN_ROOT = os.getenv("CUDNN_ROOT") or os.getenv("CUDNN_HOME") or os.getenv("CUDNN_PATH")
if CUDNN_ROOT ~= nil then
    add_includedirs(CUDNN_ROOT .. "/include")
end

target("infiniop-nvidia")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_policy("build.cuda.devlink", true)
    set_toolchains("cuda")
    add_links("cudart", "cublas")
    if has_config("cudnn") then
        add_links("cudnn")
    end
    add_cugencodes("native")

    on_load(function (target)
        import("lib.detect.find_tool")
        local nvcc = find_tool("nvcc")
        if nvcc ~= nil then
            if is_plat("windows") then
                nvcc_path = os.iorun("where nvcc"):match("(.-)\r?\n")
            else
                nvcc_path = nvcc.program
            end

            target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs")
            target:add("links", "cuda")
        end
    end)

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
        add_cuflags("--expt-relaxed-constexpr")
        if CUDNN_ROOT ~= nil then
            add_linkdirs(CUDNN_ROOT .. "/lib")
        end
    end

    add_cuflags("-Xcompiler=-Wno-error=deprecated-declarations")

    set_languages("cxx17")
    add_files("../src/infiniop/devices/nvidia/*.cu", "../src/infiniop/ops/*/nvidia/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c")
    end
target_end()

target("infinirt-nvidia")
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

target("infiniccl-nvidia")
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
