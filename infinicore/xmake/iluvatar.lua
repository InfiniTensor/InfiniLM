toolchain("iluvatar.toolchain")
    set_toolset("cc"  , "clang"  )
    set_toolset("cxx" , "clang++")
    set_toolset("cu"  , "clang++")
    set_toolset("culd", "clang++")
    set_toolset("cu-ccbin", "$(env CXX)", "$(env CC)")
toolchain_end()

rule("iluvatar.env")
    add_deps("cuda.env", {order = true})
    after_load(function (target)
        local old = target:get("syslinks")
        local new = {}

        for _, link in ipairs(old) do
            if link ~= "cudadevrt" then
                table.insert(new, link)
            end
        end

        if #old > #new then
            target:set("syslinks", new)
            local log = "cudadevrt removed, syslinks = { "
            for _, link in ipairs(new) do
                log = log .. link .. ", "
            end
            log = log:sub(0, -3) .. " }"
            print(log)
        end
    end)
rule_end()

target("infiniop-iluvatar")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("iluvatar.toolchain")
    add_rules("iluvatar.env")
    set_values("cuda.rdc", false)

    add_links("cudart", "cublas", "cudnn")

    set_warnings("all", "error")
    add_cuflags("-Wno-error=unused-private-field")
    add_cuflags("-fPIC", "-x", "ivcore", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")

    -- set_languages("cxx17") 天数似乎不能用这个配置
    add_files("../src/infiniop/devices/nvidia/*.cu", "../src/infiniop/ops/*/nvidia/*.cu")

    if has_config("ninetoothed") then
        add_files("../build/ninetoothed/*.c", {cxflags = {"-Wno-return-type"}})
    end
target_end()

target("infinirt-iluvatar")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    set_toolchains("iluvatar.toolchain")
    add_rules("iluvatar.env")
    set_values("cuda.rdc", false)

    add_links("cudart")

    set_warnings("all", "error")
    add_cuflags("-fPIC", "-x", "ivcore", "-std=c++17", {force = true})
    add_culdflags("-fPIC")
    add_cxflags("-fPIC")

    -- set_languages("cxx17") 天数似乎不能用这个配置
    add_files("../src/infinirt/cuda/*.cu")
target_end()

target("infiniccl-iluvatar")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)

    if has_config("ccl") then
        set_toolchains("iluvatar.toolchain")
        add_rules("iluvatar.env")
        set_values("cuda.rdc", false)

        add_links("cudart")

        set_warnings("all", "error")
        add_cuflags("-fPIC", "-x", "ivcore", "-std=c++17", {force = true})
        add_culdflags("-fPIC")
        add_cxflags("-fPIC")

        local nccl_root = os.getenv("NCCL_ROOT")
        if nccl_root then
            add_includedirs(nccl_root .. "/include")
            add_links(nccl_root .. "/lib/libnccl.so")
        else
            add_links("nccl") -- Fall back to default nccl linking
        end

        -- set_languages("cxx17") 天数似乎不能用这个配置
        add_files("../src/infiniccl/cuda/*.cu")
    end
target_end()
