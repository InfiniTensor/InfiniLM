local MUSA_HOME = os.getenv("MUSA_INSTALL_PATH")
add_includedirs(MUSA_HOME .. "/include")
add_linkdirs(MUSA_HOME .. "/lib")
add_links("libmusa.so")
add_links("libmusart.so")
add_links("libmudnn.so")
add_links("libmublas.so")

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

target("infini-musa")
    set_kind("static")
    set_languages("cxx17")
    add_files("../src/infiniop/devices/musa/*.cc", "../src/infiniop/ops/*/musa/*.cc")
    add_files("src/ops/*/musa/*.mu", {rule = "mu"})
    add_cxflags("-lstdc++ -Wall -fPIC")
target_end()
