
local NEUWARE_HOME = os.getenv("NEUWARE_HOME") or "/usr/local/neuware"
add_includedirs(path.join(NEUWARE_HOME, "include"))
add_linkdirs(path.join(NEUWARE_HOME, "lib64"))
add_linkdirs(path.join(NEUWARE_HOME, "lib"))
add_links("libcnrt.so")
add_links("libcnnl.so")
add_links("libcnnl_extra.so")
add_links("libcnpapi.so")

rule("mlu")
    set_extensions(".mlu")

    on_load(function (target)
        target:add("includedirs", path.join(os.projectdir(), "include"))
    end)

    on_build_file(function (target, sourcefile)
        local objectfile = target:objectfile(sourcefile)
        os.mkdir(path.directory(objectfile))

        local cc = "cncc"

        local includedirs = table.concat(target:get("includedirs"), " ")
        local args = {"-c", sourcefile, "-o", objectfile, "--bang-mlu-arch=mtp_592", "-O3", "-fPIC", "-Wall", "-Werror", "-std=c++17", "-pthread"}

        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end

        os.execv(cc, args)
        table.insert(target:objectfiles(), objectfile)
    end)
rule_end()

local src_dir = path.join(os.projectdir(), "src", "infiniop")

target("infiniop-cambricon")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    add_cxflags("-lstdc++ -fPIC")
    set_warnings("all", "error")

    set_languages("cxx17")
    add_files(src_dir.."/devices/bang/*.cc", src_dir.."/ops/*/bang/*.cc")
    local mlu_files = os.files(src_dir .. "/ops/*/bang/*.mlu")
    if #mlu_files > 0 then
        add_files(mlu_files, {rule = "mlu"})
    end
target_end()

target("infinirt-cambricon")
    set_kind("static")
    add_deps("infini-utils")
    set_languages("cxx17")
    on_install(function (target) end)
    -- Add include dirs
    add_files("../src/infinirt/bang/*.cc")
    add_cxflags("-lstdc++ -Wall -Werror -fPIC")
target_end()
