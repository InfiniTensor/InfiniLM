local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

target("infinicore_infer")
    set_kind("shared")

    -- 【关键 1】启用 CUDA 构建规则
    add_rules("cuda")

    -- 【关键 2】设置 CUDA 架构和 BF16 支持
    -- BF16 类型 (__nv_bfloat16) 需要 Compute Capability >= 8.0 (Ampere架构，如 A100, A800, 3090, 4090)
    -- 如果你的显卡较旧（如 V100/T4），这里需要改为 sm_70 或 sm_75，但可能不支持 bf16 原生指令
    add_cuflags("-arch=sm_80", "--expt-relaxed-constexpr")
    
    if is_mode("release") then
        set_optimize("fastest")
    end

    add_includedirs("include", { public = false })
    add_includedirs(INFINI_ROOT .. "/include", { public = true })

    add_linkdirs(INFINI_ROOT .. "/lib")
    add_links("infiniop", "infinirt", "infiniccl")

    -- 【关键 3】链接 CUDA Runtime 库
    add_syslinks("cudart") 

    set_languages("cxx17")
    
    -- 【调整】移除 "error"，防止 NVCC 警告导致编译中断
    set_warnings("all")

    -- 源文件添加
    add_files("src/models/*.cpp")
    add_files("src/models/*/*.cpp")
    
    -- 确保包含 Qwen3MoE 下的 C++ 和 CUDA 文件
    add_files("src/models/Qwen3MoE/*.cpp")
    add_files("src/models/Qwen3MoE/*.cu") 
    
    add_files("src/tensor/*.cpp")
    add_files("src/allocator/*.cpp")
    add_files("src/dataloader/*.cpp")
    add_files("src/cache_manager/*.cpp")
    
    add_includedirs("include")

    set_installdir(INFINI_ROOT)
    add_installfiles("include/infinicore_infer.h", {prefixdir = "include"})
    add_installfiles("include/infinicore_infer/models/*.h", {prefixdir = "include/infinicore_infer/models"})
target_end()