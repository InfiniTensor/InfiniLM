add_rules("mode.debug", "mode.release")
-- Define color codes
local GREEN = '\27[0;32m'
local YELLOW = '\27[1;33m'
local NC = '\27[0m'  -- No Color

add_includedirs("include")

if is_mode("debug") then
    add_cxflags("-g -O0")
    add_defines("DEBUG_MODE")
end

-- CPU
option("cpu")
    set_default(true)
    set_showmenu(true)
    set_description("Whether to complie implementations for CPU")
option_end()

option("omp")
    set_default(false)
    set_showmenu(true)
    set_description("Enable or disable OpenMP support for cpu kernel")
option_end()

if has_config("cpu") then
    includes("xmake/cpu.lua")
    add_defines("ENABLE_CPU_API")
end

-- 英伟达
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to complie implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_CUDA_API")
    includes("xmake/cuda.lua")
end

-- 寒武纪
option("cambricon-mlu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to complie implementations for Cambricon MLU")
option_end()

if has_config("cambricon-mlu") then
    add_defines("ENABLE_CAMBRICON_API")
    includes("xmake/bang.lua")
end

-- 华为昇腾
option("ascend-npu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to complie implementations for Huawei Ascend NPU")
option_end()

if has_config("ascend-npu") then
    add_defines("ENABLE_ASCEND_API")
    includes("xmake/ascend.lua")
end

-- 沐曦
option("metax-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to complie implementations for MetaX GPU")
option_end()

if has_config("metax-gpu") then
    add_defines("ENABLE_MACA_API")
end

-- 摩尔线程
option("moore-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to complie implementations for Moore Threads GPU")
option_end()

if has_config("mthreads-gpu") then
    add_defines("ENABLE_MUSA_API") 
end 

-- 海光
option("sugon-dcu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to complie implementations for Sugon DCU")
option_end()

if has_config("sugon-dcu") then
    add_defines("ENABLE_CUDA_API")
    add_defines("ENABLE_SUGON_CUDA_API")
end


target("infiniop")
    set_kind("shared")

    if has_config("cpu") then
        add_deps("infiniop-cpu")
    end
    if has_config("nv-gpu") then
        add_deps("infiniop-cuda")
    end
    if has_config("sugon-dcu") then
        local builddir = string.format(
            "build/%s/%s/%s",
            get_config("plat"),
            get_config("arch"),
            get_config("mode")
        )
        add_shflags("-s", "-shared", "-fPIC")
        add_links("cublas", "cudnn", "cudadevrt", "cudart_static", "rt", "pthread", "dl")
        -- Using -linfiniop-cuda will fail, manually link the target using full path
        add_deps("nv-gpu", {inherit = false})
        add_links(builddir.."/libinfiniop-cuda.a")
        set_toolchains("sugon-dcu-linker")
    end

    if has_config("cambricon-mlu") then
        add_deps("infiniop-cambricon")
    end
    if has_config("ascend-npu") then
        add_deps("infiniop-ascend")
    end
    if has_config("metax-gpu") then
        add_deps("metax-gpu")
    end
    set_languages("cxx17")
    add_files("src/infiniop/devices/handle.cc")
    add_files("src/infiniop/ops/*/operator.cc")
    add_files("src/infiniop/*.cc")
    after_build(function (target) print(YELLOW .. "You can install the libraries with \"xmake install\"" .. NC) end)

    set_installdir(os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini"))
    add_installfiles("include/infiniop/(**/*.h)", {prefixdir = "include/infiniop"})
    add_installfiles("include/infiniop/*.h", {prefixdir = "include/infiniop"})
    add_installfiles("include/infiniop.h", {prefixdir = "include"})
    add_installfiles("include/infinicore.h", {prefixdir = "include"})
target_end()
