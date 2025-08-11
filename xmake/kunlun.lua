add_defines("ENABLE_KUNLUN_API")
local KUNLUN_HOME = os.getenv("KUNLUN_HOME")
local XRE_DIR = path.join(KUNLUN_HOME, "xre")
local XTDK_DIR = path.join(KUNLUN_HOME, "xtdk")
local XDNN_DIR = path.join(KUNLUN_HOME, "xhpc", "xdnn")

-- Add include dirs
add_includedirs(path.join(XRE_DIR, "include"))
add_includedirs(path.join(XDNN_DIR, "include"))
add_includedirs(path.join(XTDK_DIR, "include"))
add_linkdirs(path.join(XRE_DIR, "so"))
add_linkdirs(path.join(XDNN_DIR, "so"))
add_links("xpurt", "xpuapi")

rule("xpu")
    set_extensions(".xpu")
    
    on_build_file(function (target, sourcefile)

        local sourcefile_config = target:fileconfig(sourcefile) or {}
        local includedirs = sourcefile_config.includedirs or {}

        local objectfile = target:objectfile(sourcefile)
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
            "--xpu-arch=xpu3",
            "-std=c++17",
            "-O2",
            "-fno-builtin",
            "-c", sourcefile,
            "-o", objectfile
        }
        
        for _, includedir in ipairs(target:get("includedirs")) do
            table.insert(args, "-I" .. includedir)
        end

        -- print(args)
        local ok, code = os.execv(cc, args)
        assert(ok == 0, "Compile failed: " .. sourcefile)

        table.insert(target:objectfiles(), objectfile)
        print(target:objectfiles())
    end)
rule_end()

local src_dir = path.join(os.projectdir(), "src", "infiniop")

target("infiniop-kunlun")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    add_cxflags("-lstdc++ -fPIC -Wno-error=unused-function")
    set_warnings("all", "error")

    set_languages("cxx17")
    add_files("$(projectdir)/src/infiniop/devices/kunlun/*.cc", "$(projectdir)/src/infiniop/ops/*/kunlun/*.cc")
    -- compile handwriting kernel
    local xpu_files = os.files(src_dir .. "/ops/*/kunlun/*.xpu")
    if #xpu_files > 0 then
        add_files(xpu_files, {
            rule = "xpu",
            includedirs = {
                path.join(os.projectdir, "include"),
                path.join(XRE_DIR, "include"),
                path.join(XDNN_DIR, "include"),
                path.join(XTDK_DIR, "include")
            }
        })
    end
target_end()

target("infinirt-kunlun")
    set_kind("static")
    add_deps("infini-utils")
    set_languages("cxx17")
    on_install(function (target) end)
    -- Add include dirs
    add_files("$(projectdir)/src/infinirt/kunlun/*.cc")
    add_cxflags("-lstdc++ -Wall -Werror -fPIC")

target_end()
