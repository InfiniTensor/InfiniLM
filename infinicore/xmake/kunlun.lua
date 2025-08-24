add_defines("ENABLE_KUNLUN_API")
local KUNLUN_HOME = os.getenv("KUNLUN_HOME")
local XTDK_DIR = path.join(KUNLUN_HOME, "XTDK")

-- Add include dirs
add_includedirs(path.join(KUNLUN_HOME, "include"), {public=true})
add_linkdirs(path.join(KUNLUN_HOME, "lib64"))
add_links("xpurt")
add_links("xpuapi")

rule("xpu")
    set_extensions(".xpu")
    
    on_load(function (target)
        target:add("includedirs", path.join(os.projectdir(), "include"))
    end)

    on_build_file(function (target, sourcefile)

        local objectfile = target:objectfile(sourcefile)
        local basename = objectfile:gsub("%.o$", "")
        os.mkdir(path.directory(objectfile))
        local cc = path.join(XTDK_DIR, "bin/clang++")
        local includedirs = table.concat(target:get("includedirs"), " ")
        local arch_map = {
            ["x86_64"] = "x86_64-linux-gnu",
            ["arm64"] = "aarch64-linux-gnu"
        }


        local args = {
            "--sysroot=/",
            "--target=" .. arch_map[os.arch()],
            "-fPIC",
            "-pie",
            "--xpu-arch=xpu2",
            "--basename", basename,
            "-std=c++11",
            "-O2",
            "-fno-builtin",
            "-g",
            "-c", sourcefile,
            "-v"
        }
        
        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end

        -- print(args)
        os.execv(cc, args)
        table.insert(target:objectfiles(), objectfile)
        table.insert(target:objectfiles(), basename .. ".device.bin.o")
        print(target:objectfiles())
    end)
rule_end()

local src_dir = path.join(os.projectdir(), "src", "infiniop")

target("infiniop-kunlun")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    add_cxflags("-lstdc++ -fPIC")
    set_warnings("all", "error")

    set_languages("cxx17")
    add_files("$(projectdir)/src/infiniop/devices/kunlun/*.cc", "$(projectdir)/src/infiniop/ops/*/kunlun/*.cc")
    -- compile handwriting kernel
    local xpu_files = os.files(src_dir .. "/ops/*/kunlun/*.xpu")
    if #xpu_files > 0 then
        add_files(xpu_files, {rule = "xpu"})
    end
target_end()

target("infinirt-kunlun")
    set_kind("static")
    add_deps("infini-utils")
    set_languages("cxx17")
    on_install(function (target) end)
    -- Add include dirs
    add_files("../src/infinirt/kunlun/*.cc")
    add_cxflags("-lstdc++ -Wall -Werror -fPIC")

target_end()
