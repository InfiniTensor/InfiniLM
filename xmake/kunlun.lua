add_defines("ENABLE_KUNLUN_API")
local KUNLUN_HOME = os.getenv("KUNLUN_HOME")

-- Add include dirs
add_includedirs(path.join(KUNLUN_HOME, "include"), {public=true})
add_linkdirs(path.join(KUNLUN_HOME, "lib64"))
add_links("xpurt")
add_links("xpuapi")

target("infiniop-kunlun")
    set_kind("static")
    add_deps("infini-utils")
    on_install(function (target) end)

    add_cxflags("-lstdc++ -fPIC")
    set_warnings("all", "error")

    set_languages("cxx17")
    add_files("$(projectdir)/src/infiniop/devices/kunlun/*.cc", "$(projectdir)/src/infiniop/ops/*/kunlun/*.cc")
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
