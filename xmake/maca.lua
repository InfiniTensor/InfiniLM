
local MACA_ROOT = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or os.getenv("MACA_ROOT")
add_includedirs(MACA_ROOT .. "/include")
add_linkdirs(MACA_ROOT .. "/lib")
add_links("hcdnn", "hcblas", "hcruntime")

rule("maca")
    set_extensions(".maca")

    on_load(function (target)
        target:add("includedirs", "include")
    end)

    on_build_file(function (target, sourcefile)
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))
        local htcc = path.join(MACA_ROOT, "htgpu_llvm/bin/htcc")
        local includedirs = table.concat(target:get("includedirs"), " ")

        local args = { "-x", "hpcc", "-c", sourcefile, "-o", objectfile, "-I" .. MACA_ROOT .. "/include", "-O3", "-fPIC", "-Werror", "-std=c++17"}

        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end

        os.execv(htcc, args)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("infiniop-metax")
    set_kind("static")
    on_install(function (target) end)
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cxflags("-lstdc++", "-fPIC", "-Wno-defaulted-function-deleted", "-Wno-strict-aliasing")
    add_files("../src/infiniop/devices/maca/*.cc", "../src/infiniop/ops/*/maca/*.cc")
    add_files("../src/infiniop/ops/*/maca/*.maca", {rule = "maca"})
target_end()

target("infinirt-metax")
    set_kind("static")
    set_languages("cxx17")
    on_install(function (target) end)
    add_deps("infini-utils")
    set_warnings("all", "error")
    add_cxflags("-lstdc++ -fPIC")
    add_files("../src/infinirt/maca/*.cc")
target_end()

target("infiniccl-metax")
    set_kind("static")
    add_deps("infinirt")
    on_install(function (target) end)
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC")
    end
    if has_config("ccl") then
        add_links("libhccl.so")
        add_files("../src/infiniccl/maca/*.cc")
    end
    set_languages("cxx17")
    
target_end()
