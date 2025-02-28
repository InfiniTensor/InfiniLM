add_defines("ENABLE_KUNLUN_API")
local KUNLUN_HOME = os.getenv("KUNLUN_HOME")

-- Add include dirs
add_includedirs(path.join(KUNLUN_HOME, "include"), {public=true})
add_linkdirs(path.join(KUNLUN_HOME, "lib64"))
add_links("xpurt")
add_links("xpuapi")

target("infiniop-kunlun")
    -- Other configs
    set_kind("static")
    set_languages("cxx17")
    on_install(function (target) end)
    -- Add files
    add_files("$(projectdir)/src/infiniop/devices/kunlun/*.cc", "$(projectdir)/src/infiniop/ops/*/kunlun/*.cc")
    add_cxflags("-lstdc++ -Wall -Werror -fPIC")

target_end()
