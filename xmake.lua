add_requires("pybind11")

local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")
local INFINI_RT_ROOT = os.getenv("INFINI_RT_ROOT")

local function validate_external_infinirt(xmake_os)
    if not INFINI_RT_ROOT or INFINI_RT_ROOT == "" then
        return
    end

    local public_header = path.join(INFINI_RT_ROOT, "include", "infini", "rt.h")
    if not xmake_os.isfile(public_header) then
        xmake_os.raise("InfiniRT public header not found: " .. public_header)
    end

    local library = path.join(INFINI_RT_ROOT, "lib", "libinfinirt.so")
    if not xmake_os.isfile(library) then
        library = path.join(INFINI_RT_ROOT, "lib64", "libinfinirt.so")
    end
    if not xmake_os.isfile(library) then
        xmake_os.raise("InfiniRT library not found under: " .. INFINI_RT_ROOT)
    end
end

local function add_external_infinirt()
    if not INFINI_RT_ROOT or INFINI_RT_ROOT == "" then
        return
    end
    add_includedirs(INFINI_RT_ROOT.."/include")
    for _, libdir in ipairs({INFINI_RT_ROOT.."/lib", INFINI_RT_ROOT.."/lib64"}) do
        if os.isdir(libdir) then
            add_linkdirs(libdir)
            add_rpathdirs(libdir)
        end
    end
end

set_toolchains("gcc")

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/json/single_include/")

target("infinicore_infer")
    set_kind("shared")
    on_load(function (target)
        validate_external_infinirt(os)
    end)

    add_includedirs("include", { public = false })
    add_external_infinirt()
    add_includedirs(INFINI_ROOT.."/include", { public = true })

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    set_languages("cxx17")
    set_warnings("all", "error")

    add_files("src/models/*.cpp")
    add_files("src/models/*/*.cpp")
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    add_files("src/dataloader/*.cpp")
    add_files("src/cache_manager/*.cpp")
    add_includedirs("include")

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
target_end()

target("_infinilm")
    add_packages("pybind11")
    set_default(false)
    add_rules("python.module", {soabi = true})
    set_languages("cxx17")
    set_kind("shared")
    on_load(function (target)
        validate_external_infinirt(os)
    end)

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_external_infinirt()
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    add_includedirs("include", { public = false })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")

    -- Add src files
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")

    set_installdir("python/infinilm")
target_end()
