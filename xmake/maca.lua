
local MACA_ROOT = os.getenv("MACA_PATH") or os.getenv("MACA_HOME") or os.getenv("MACA_ROOT")

add_includedirs(MACA_ROOT .. "/include")
add_linkdirs(MACA_ROOT .. "/lib")
add_links("libhcdnn.so")
add_links("libhcblas.so")
add_links("libhcruntime.so")

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
    add_cxflags("-lstdc++ -Wall -fPIC")
    set_languages("cxx17")
    set_warnings("all")

    add_files("../src/infiniop/devices/maca/*.cc", "../src/infiniop/ops/*/maca/*.cc")
    add_files("../src/infiniop/ops/*/maca/*.maca", {rule = "maca"})

target_end()
