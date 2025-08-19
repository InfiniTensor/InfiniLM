local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

target("infinicore_infer")
    set_kind("shared")

    -- [[--- 使用旧版语法来支持调试 ---]]
    -- 这会为你的目标应用 "debug" 和 "release" 模式
    -- "debug" 模式会自动设置 -g 和 -O0
    -- "release" 模式会自动设置 -O2/O3 和剥离符号
    add_rules("mode.debug", "mode.release")
    -- [[-----------------------------------]]
	add_cxflags("-Wno-error")
    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("src/models/*/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    add_includedirs("include")

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
	--add_cxflags("-fsanitize=address", "-fno-omit-frame-pointer", "-g")
    --add_ldflags("-fsanitize=address")
target_end()