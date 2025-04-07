local MUSA_HOME = os.getenv("MUSA_INSTALL_PATH")
add_includedirs(MUSA_HOME .. "/include")
add_linkdirs(MUSA_HOME .. "/lib")
add_links("musa", "musart", "mudnn", "mublas")

rule("mu")
    set_extensions(".mu")
    on_load(function (target)
        target:add("includedirs", "include")
    end)

    on_build_file(function (target, sourcefile)
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))

        local mcc = MUSA_HOME .. "/bin/mcc"
        local includedirs = table.concat(target:get("includedirs"), " ")
        local args = {"-c", sourcefile, "-o", objectfile, "-I" .. MUSA_HOME .. "/include", "-O3", "-fPIC", "-Wall", "-std=c++17", "-pthread"}
        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end

        os.execv(mcc, args)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

target("infiniop-moore")
    set_kind("static")
    on_install(function (target) end)
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cxflags("-lstdc++", "-fPIC", "-Wno-comment")
    add_files("../src/infiniop/devices/musa/*.cc", "../src/infiniop/ops/*/musa/*.cc")
    add_files("../src/infiniop/ops/*/musa/*.mu", {rule = "mu"})
target_end()

target("infinirt-moore")
    set_kind("static")
    set_languages("cxx17")
    on_install(function (target) end)
    add_deps("infini-utils")
    set_warnings("all", "error")
    add_cxflags("-lstdc++", "-fPIC")
    add_files("../src/infinirt/musa/*.cc")
target_end()
